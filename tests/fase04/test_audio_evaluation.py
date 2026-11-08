from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "entregas" / "fase04" / "src"))

import evaluate_audio_flags as harness  # noqa: E402


def test_parse_cycle_annotations_skips_header_and_keeps_flags(tmp_path):
    path = tmp_path / "101_1b1_Al_sc_Meditron.txt"
    path.write_text(
        "start\tend\tcrackle\twheeze\n0.5\t1.5\t1\t0\n1.5\t2.5\t0\t1\n2.5\t3.5\t0\t0\n",
        encoding="utf-8",
    )

    assert harness.parse_cycle_annotations(path) == [(1, 0), (0, 1), (0, 0)]


def test_load_diagnosis_parses_tab_separated_pairs(tmp_path):
    path = tmp_path / "ICBHI_Challenge_diagnosis.txt"
    path.write_text("101\tURTI\n102\tHealthy\n", encoding="utf-8")

    assert harness.load_diagnosis(path) == {"101": "URTI", "102": "Healthy"}


def test_confusion_computes_precision_recall_and_f1():
    labels = [True, True, False, False]
    predictions = [True, False, True, False]

    metrics = harness._confusion(labels, predictions)

    assert metrics["true_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["accuracy"] == 0.5


def test_always_true_baseline_matches_positive_rate():
    labels = [True, True, False, False]
    predictions = [True] * len(labels)

    metrics = harness._confusion(labels, predictions)

    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["accuracy"] == 0.5


def test_decimate_halves_length_and_keeps_finite_values():
    signal = np.sin(np.linspace(0, 2 * np.pi * 100, 8000)).astype(np.float32)

    reduced, rate = harness.decimate(signal, 8000, 4000)

    assert rate == 4000
    assert reduced.size == 4000
    assert np.all(np.isfinite(reduced))


def test_decimate_is_noop_when_already_at_target_rate():
    signal = np.ones(100, dtype=np.float32)

    reduced, rate = harness.decimate(signal, 4000, 8000)

    assert rate == 4000
    assert reduced is signal


def test_parse_wfdb_header_reads_rate_and_count(tmp_path):
    header = tmp_path / "voice001.hea"
    header.write_text("voice001 1 8000 40000\nvoice001.dat 16 1.0\n", encoding="utf-8")

    assert harness.parse_wfdb_header(header) == (8000, 40000)


def test_parse_wfdb_header_rejects_short_line(tmp_path):
    header = tmp_path / "voice001.hea"
    header.write_text("voice001 1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        harness.parse_wfdb_header(header)


def test_load_voiced_diagnosis_extracts_field(tmp_path):
    info = tmp_path / "voice001-info.txt"
    info.write_text(
        "Name: voice001\nDiagnosis: hyperkinetic dysphonia\nSession: 1\n",
        encoding="utf-8",
    )

    assert harness.load_voiced_diagnosis(info) == "hyperkinetic dysphonia"


def test_load_voiced_diagnosis_returns_none_when_absent(tmp_path):
    info = tmp_path / "voice001-info.txt"
    info.write_text("Name: voice001\n", encoding="utf-8")

    assert harness.load_voiced_diagnosis(info) is None


def test_iter_voiced_recordings_pairs_signal_header_and_diagnosis(tmp_path):
    (tmp_path / "voice001-info.txt").write_text(
        "Diagnosis: Healthy\n", encoding="utf-8"
    )
    (tmp_path / "voice001.hea").write_text(
        "voice001 1 8000 40000\n", encoding="utf-8"
    )
    (tmp_path / "voice001.txt").write_text("0.0\n0.1\n", encoding="utf-8")
    (tmp_path / "voice002-info.txt").write_text("Diagnosis: Healthy\n", encoding="utf-8")

    recordings = harness.iter_voiced_recordings(tmp_path)

    assert [path.name for path, _header, _diag in recordings] == ["voice001.txt"]
    assert recordings[0][2] == "Healthy"


def test_load_voiced_metrics_returns_feature_keys(tmp_path):
    sample_rate = 8000
    samples = 0.4 * np.sin(
        2 * np.pi * 220 * np.arange(sample_rate) / sample_rate
    ).astype(np.float32)
    signal_path = tmp_path / "voice001.txt"
    np.savetxt(signal_path, samples)
    header_path = tmp_path / "voice001.hea"
    header_path.write_text(f"voice001 1 {sample_rate} {samples.size}\n", encoding="utf-8")

    metrics = harness.load_voiced_metrics(signal_path, header_path, sample_rate, 5.0)

    expected_keys = set(harness.FEATURE_KEYS) | set(harness.DYSARTHRIA_FEATURE_KEYS)
    assert set(metrics) == expected_keys
    assert metrics["duration_seconds"] > 0.0
    assert metrics["rms_energy"] > 0.0


def _parkinson_csv(path):
    path.write_text(
        "name,MDVP:Jitter(%),MDVP:Shimmer,HNR,status\n"
        "phon_R01_S01_1,0.004,0.02,25.0,0\n"
        "phon_R01_S02_1,0.005,0.03,24.0,0\n"
        "phon_R01_S03_1,0.02,0.09,12.0,1\n"
        "phon_R01_S04_1,0.03,0.1,10.0,1\n",
        encoding="utf-8",
    )
    return path


def test_load_parkinson_rows_maps_voice_measures(tmp_path):
    data = _parkinson_csv(tmp_path / "parkinsons.data")

    rows = harness.load_parkinson_rows(data)

    assert rows[0][0] == "phon_R01_S01"
    assert rows[0][1] == {"jitter_local": 0.4, "shimmer_local": 2.0, "hnr": 25.0}
    assert rows[0][2] is False
    assert rows[-1][2] is True


def test_evaluate_parkinson_reports_dataset_and_rules(tmp_path):
    data = _parkinson_csv(tmp_path / "parkinsons.data")

    report = harness.evaluate_parkinson(data, limit=None)

    assert report["dataset"]["recordings_used"] == 4
    assert report["dataset"]["recordings_pathological"] == 2
    assert report["proxy_label_primary"]["name"] == "parkinson_diagnosis"
    assert "possivel_disartria" in report["proxy_label_primary"]["rule_based"]


def test_confusion_reports_sensitivity_and_specificity():
    labels = [True, True, False, False]
    predictions = [True, False, False, False]

    metrics = harness._confusion(labels, predictions)

    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 1.0


def test_bootstrap_interval_returns_ordered_bounds():
    labels = [True] * 20 + [False] * 20
    scores = [0.9] * 20 + [0.1] * 20

    interval = harness._bootstrap_interval(
        labels, scores, harness._f1_from_scores, rounds=200
    )

    assert interval["resamples"] > 0
    assert interval["lower"] is not None
    assert interval["lower"] <= interval["upper"]


def test_calibrate_limits_uses_healthy_percentiles():
    healthy = [
        {"jitter_local": 1.0, "shimmer_local": 2.0, "hnr": 30.0},
        {"jitter_local": 2.0, "shimmer_local": 3.0, "hnr": 28.0},
        {"jitter_local": 3.0, "shimmer_local": 4.0, "hnr": 26.0},
        {"jitter_local": 4.0, "shimmer_local": 5.0, "hnr": 24.0},
        {"jitter_local": 5.0, "shimmer_local": 6.0, "hnr": 22.0},
    ]

    limits = harness.calibrate_voice_quality_limits(healthy)

    assert limits == {"jitter_local": 4.8, "shimmer_local": 5.8, "hnr": 22.4}


def test_calibrate_limits_falls_back_to_defaults_without_healthy_rows():
    assert harness.calibrate_voice_quality_limits([]) == dict(harness.DEFAULT_LIMITS)


def test_cross_transfer_scores_the_target_dataset():
    rng = np.random.default_rng(0)
    width = len(harness.FEATURE_KEYS)
    source = rng.normal(size=(40, width))
    source[:20, 0] -= 2.0
    source[20:, 0] += 2.0
    source_labels = [index < 20 for index in range(40)]
    target = rng.normal(size=(30, width))
    target[:15, 0] -= 2.0
    target[15:, 0] += 2.0
    target_labels = [index < 15 for index in range(30)]

    metrics = harness._cross_transfer(source, source_labels, target, target_labels)

    assert metrics["roc_auc"] is not None
    assert metrics["roc_auc"] >= 0.8
    assert metrics["average_precision"] is not None


def test_evaluate_parkinson_reports_calibrated_dysarthria(tmp_path):
    data = _parkinson_csv(tmp_path / "parkinsons.data")

    report = harness.evaluate_parkinson(data, limit=None)

    calibrated = report["proxy_label_primary"]["dysarthria_calibrated"]
    assert set(calibrated["limits"]) == {"jitter_local", "shimmer_local", "hnr"}
    assert calibrated["default_limits"] == dict(harness.DEFAULT_LIMITS)
    assert "possivel_disartria" in calibrated["rule_based"]


