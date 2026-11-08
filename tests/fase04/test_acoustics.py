import numpy as np
import pytest

from backend.rehab_audio.acoustics import (
    acoustic_findings,
    extract_acoustic_features,
)


METRIC_KEYS = {
    "duration_seconds",
    "rms_energy",
    "voiced_ratio",
    "pause_ratio",
    "mean_pause_seconds",
    "speaking_rate",
    "pitch_variability",
    "spectral_centroid_hz",
    "spectral_rolloff_hz",
    "spectral_flux",
    "zero_crossing_rate",
    "jitter_local",
    "shimmer_local",
    "hnr",
}


def test_extract_acoustic_features_on_silence():
    metrics = extract_acoustic_features(np.zeros(16000, dtype=np.float32), 16000)

    assert set(metrics) == METRIC_KEYS
    assert metrics["rms_energy"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["pause_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["voiced_ratio"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["spectral_centroid_hz"] == 0.0
    assert metrics["spectral_rolloff_hz"] == 0.0
    assert metrics["spectral_flux"] == 0.0
    assert metrics["zero_crossing_rate"] == 0.0
    assert metrics["jitter_local"] == 0.0
    assert metrics["shimmer_local"] == 0.0
    assert metrics["hnr"] == 0.0


def test_extract_acoustic_features_never_raises_on_empty_signal():
    metrics = extract_acoustic_features(np.array([], dtype=np.float32), 16000)

    assert set(metrics) == METRIC_KEYS
    assert metrics["duration_seconds"] == 0.0


def test_silence_reports_insufficient_data():
    metrics = extract_acoustic_features(np.zeros(16000, dtype=np.float32), 16000)

    findings = acoustic_findings(metrics)

    assert [item["code"] for item in findings] == ["dados_insuficientes"]
    assert findings[0]["severity"] == "info"


def test_low_energy_and_long_pauses_flag_fatigue_and_breathing():
    metrics = {
        "duration_seconds": 30.0,
        "rms_energy": 0.01,
        "voiced_ratio": 0.3,
        "pause_ratio": 0.7,
        "mean_pause_seconds": 1.5,
        "speaking_rate": 0.5,
        "pitch_variability": 0.1,
    }

    codes = {item["code"] for item in acoustic_findings(metrics)}

    assert "possivel_fadiga" in codes
    assert "possivel_desconforto_respiratorio" in codes


def test_high_pitch_variability_flags_unstable_speech():
    metrics = {
        "duration_seconds": 30.0,
        "rms_energy": 0.05,
        "voiced_ratio": 0.9,
        "pause_ratio": 0.1,
        "mean_pause_seconds": 0.0,
        "speaking_rate": 2.0,
        "pitch_variability": 0.5,
    }

    codes = {item["code"] for item in acoustic_findings(metrics)}

    assert codes == {"fala_instavel"}


def test_deviant_voice_quality_flags_possible_dysarthria():
    metrics = {
        "duration_seconds": 30.0,
        "rms_energy": 0.05,
        "voiced_ratio": 0.9,
        "pause_ratio": 0.1,
        "mean_pause_seconds": 0.0,
        "speaking_rate": 2.0,
        "pitch_variability": 0.1,
        "jitter_local": 2.0,
        "shimmer_local": 5.0,
        "hnr": 12.0,
    }

    codes = {item["code"] for item in acoustic_findings(metrics)}

    assert "possivel_disartria" in codes


def test_tone_voice_quality_markers_are_finite_and_non_negative():
    sample_rate = 16000
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate)

    metrics = extract_acoustic_features(tone.astype(np.float32), sample_rate)

    for key in ("jitter_local", "shimmer_local", "hnr"):
        assert metrics[key] >= 0.0


def test_tone_spectral_centroid_tracks_fundamental_frequency():
    sample_rate = 16000
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate)

    metrics = extract_acoustic_features(tone.astype(np.float32), sample_rate)

    assert metrics["spectral_centroid_hz"] == pytest.approx(220.0, abs=5.0)
    assert 0.0 <= metrics["zero_crossing_rate"] <= 1.0
