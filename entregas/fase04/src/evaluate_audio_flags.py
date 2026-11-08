"""Evaluate the rehabilitation audio acoustic flags on public voice datasets.

Offline evaluation harness for Tarefa 2 (analise de audio em reabilitacao). It
serves two datasets:

* ``icbhi`` - ICBHI 2017 Respiratory Sound Database. The flags were written for
  speech recorded during a physiotherapy session, while ICBHI contains lung
  sounds, so this experiment measures something narrower: whether the acoustic
  layer carries any signal for respiratory abnormality. It does NOT validate
  fatigue detection, because no public dataset labels fatigue in rehabilitation
  sessions. Proxy label: a recording is abnormal when at least one annotated
  respiratory cycle contains crackles or wheezes.

* ``voiced`` - VOICED Database (PhysioNet). 208 sustained-vowel recordings with
  a clinically verified healthy/pathological voice label. This is the labeled
  *voice* dataset that best matches the "alteracoes vocais" claim: it exercises
  the pitch/energy features on the same construct. It cannot validate the pause
  based flags (a sustained vowel has no pauses) and remains a proxy for
  rehabilitation speech.

* ``parkinson`` - UCI Parkinson's Disease (open, CC BY 4.0). Sustained-vowel
  voice measures (jitter/shimmer/HNR) labeled healthy vs Parkinson's. It is the
  open proxy that sanity-checks the dysarthria markers, but it ships precomputed
  MDVP measures rather than raw audio.

Running ``--dataset voiced`` also trains and persists the reusable classifier
consumed by ``backend/rehab_audio/voice_model.py``.

The optional ``--include-embeddings`` flag adds a deep-embedding probe: the
``icbhi`` and ``voiced`` experiments then compare a linear classifier over
wav2vec2 mean-pooled embeddings against the handcrafted-feature baseline. The
flag needs ``torch``/``transformers``; without them the probe is reported as
skipped and the rest of the evaluation is unaffected.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.rehab_audio.acoustics import (
    FEATURE_KEYS,
    FEATURE_VERSION,
    acoustic_findings,
    decimate,
    extract_acoustic_features,
    to_mono,
)
from backend.rehab_audio.dysarthria import (
    DEFAULT_LIMITS,
    DYSARTHRIA_FEATURE_KEYS,
    dysarthria_findings,
)
from backend.rehab_audio.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_ID,
    EMBEDDING_VERSION,
    embeddings_available,
    extract_embeddings,
)

DATA_ROOT = PROJECT_ROOT / "entregas" / "fase04" / "data" / "raw" / "icbhi"
REPORT_PATH = (
    PROJECT_ROOT / "entregas" / "fase04" / "results" / "audio_flags_evaluation.json"
)

VOICED_DATA_ROOT = (
    PROJECT_ROOT
    / "entregas"
    / "fase04"
    / "data"
    / "raw"
    / "voiced"
    / "voice-icar-federico-ii-database-1.0.0"
)
VOICED_REPORT_PATH = (
    PROJECT_ROOT / "entregas" / "fase04" / "results" / "voice_flags_evaluation.json"
)
VOICED_MODEL_PATH = (
    PROJECT_ROOT / "backend" / "model" / "voice_alteration_classifier.joblib"
)

PARKINSON_DATA_PATH = (
    PROJECT_ROOT / "entregas" / "fase04" / "data" / "raw" / "parkinson" / "parkinsons.data"
)
PARKINSON_REPORT_PATH = (
    PROJECT_ROOT / "entregas" / "fase04" / "results" / "dysarthria_evaluation.json"
)
CROSS_REPORT_PATH = (
    PROJECT_ROOT / "entregas" / "fase04" / "results" / "audio_cross_evaluation.json"
)

WARNING_CODES = (
    "possivel_fadiga",
    "possivel_desconforto_respiratorio",
    "fala_instavel",
)
DEFAULT_TARGET_SAMPLE_RATE = 8000
DEFAULT_MAX_SECONDS = 30.0
CV_FOLDS = 5
RANDOM_STATE = 42
BOOTSTRAP_ROUNDS = 1000
CALIBRATION_LOW_PERCENTILE = 5
CALIBRATION_HIGH_PERCENTILE = 95

EMBEDDING_FEATURE_LABEL = f"{EMBEDDING_MODEL_ID} mean-pooled ({EMBEDDING_DIM}-d)"
# A regularised linear probe suits the high-dimensional, low-sample embedding
# matrix better than the boosted trees used for the handcrafted features.
EMBEDDING_MODEL_NAME = "logistic"

DATASET_CITATION = (
    "Rocha BM et al. (2019) An open access database for the evaluation of "
    "respiratory sound classification algorithms. Physiological Measurement 40 035001"
)
DATASET_LICENSE = (
    "Freely available for research. Cite the publication above and the ICBHI 2017 "
    "challenge when publishing results derived from the data."
)

VOICED_DATASET_CITATION = (
    "Cesari U, De Pietro G, Marciano E, Niri C, Sannino G, Verde L (2018) A new "
    "database of healthy and pathological voices. Computers & Electrical "
    "Engineering 68:310-321"
)
VOICED_DATASET_LICENSE = (
    "Open Data Commons Attribution License v1.0 (PhysioNet). Cite the publication "
    "and the PhysioNet resource when publishing results derived from the data."
)

PARKINSON_DATASET_CITATION = (
    "Little MA, McSharry PE, Hunter EJ, Spielman J, Ramig LO (2007) Suitability of "
    "dysphonia measurements for telemonitoring of Parkinson's disease. IEEE "
    "Transactions on Biomedical Engineering 56(4):1015-1022 (UCI ML Repository, "
    "'Parkinson's Disease', ID 174)"
)
PARKINSON_DATASET_LICENSE = (
    "Creative Commons Attribution 4.0 International (CC BY 4.0), UCI Machine "
    "Learning Repository."
)
DYSARTHRIA_CODES = ("possivel_disartria",)


def load_diagnosis(path: Path) -> dict[str, str]:
    """Read ``patient_id <tab> diagnosis`` into a mapping."""
    diagnosis: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            diagnosis[parts[0]] = parts[1]
    return diagnosis


def parse_cycle_annotations(path: Path) -> list[tuple[int, int]]:
    """Return ``(crackle, wheeze)`` flags for each annotated respiratory cycle."""
    cycles: list[tuple[int, int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            crackle = int(float(parts[2]))
            wheeze = int(float(parts[3]))
        except ValueError:
            continue
        cycles.append((crackle, wheeze))
    return cycles


def iter_recordings(data_root: Path) -> list[tuple[Path, Path]]:
    """Pair every ``.wav`` with its sibling annotation ``.txt``."""
    pairs: list[tuple[Path, Path]] = []
    for wav_path in sorted(data_root.rglob("*.wav")):
        annotation_path = wav_path.with_suffix(".txt")
        if annotation_path.is_file():
            pairs.append((wav_path, annotation_path))
    return pairs


def _metrics_from_signal(
    samples: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
    max_seconds: float,
) -> dict:
    signal = to_mono(samples)
    if max_seconds > 0 and sample_rate > 0:
        limit = int(max_seconds * sample_rate)
        if 0 < limit < signal.size:
            signal = signal[:limit]
    signal, sample_rate = decimate(signal, int(sample_rate), target_sample_rate)
    return extract_acoustic_features(signal, int(sample_rate))


def load_metrics(
    wav_path: Path,
    target_sample_rate: int,
    max_seconds: float,
) -> dict:
    samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    return _metrics_from_signal(samples, int(sample_rate), target_sample_rate, max_seconds)


def parse_wfdb_header(header_path: Path) -> tuple[int, int]:
    """Return ``(sample_rate, sample_count)`` from a WFDB header line."""
    first_line = header_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    parts = first_line.split()
    if len(parts) < 4:
        raise ValueError(f"Unexpected WFDB header in {header_path}: {first_line!r}")
    return int(parts[2]), int(parts[3])


def load_voiced_diagnosis(info_path: Path) -> str | None:
    """Return the ``Diagnosis`` field of a VOICED ``-info.txt`` file."""
    text = info_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^Diagnosis:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def iter_voiced_recordings(voiced_root: Path) -> list[tuple[Path, Path, str]]:
    """Pair each VOICED subject with its signal ``.txt`` and diagnosis."""
    recordings: list[tuple[Path, Path, str]] = []
    for info_path in sorted(voiced_root.glob("*-info.txt")):
        record = info_path.name[: -len("-info.txt")]
        signal_path = voiced_root / f"{record}.txt"
        header_path = voiced_root / f"{record}.hea"
        diagnosis = load_voiced_diagnosis(info_path)
        if signal_path.is_file() and header_path.is_file() and diagnosis:
            recordings.append((signal_path, header_path, diagnosis))
    return recordings


def load_voiced_metrics(
    signal_path: Path,
    header_path: Path,
    target_sample_rate: int,
    max_seconds: float,
) -> dict:
    sample_rate, _ = parse_wfdb_header(header_path)
    samples = np.loadtxt(str(signal_path), dtype="float32")
    return _metrics_from_signal(samples, sample_rate, target_sample_rate, max_seconds)


def _embedding_from_signal(
    samples: np.ndarray,
    sample_rate: int,
    max_seconds: float,
    model_id: str,
    cache_dir: str | None,
) -> np.ndarray | None:
    signal = to_mono(samples)
    if max_seconds > 0 and sample_rate > 0:
        limit = int(max_seconds * sample_rate)
        if 0 < limit < signal.size:
            signal = signal[:limit]
    return extract_embeddings(signal, int(sample_rate), model_id=model_id, cache_dir=cache_dir)


def load_embedding(
    wav_path: Path,
    max_seconds: float,
    model_id: str = EMBEDDING_MODEL_ID,
    cache_dir: str | None = None,
) -> np.ndarray | None:
    samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    return _embedding_from_signal(samples, int(sample_rate), max_seconds, model_id, cache_dir)


def load_voiced_embedding(
    signal_path: Path,
    header_path: Path,
    max_seconds: float,
    model_id: str = EMBEDDING_MODEL_ID,
    cache_dir: str | None = None,
) -> np.ndarray | None:
    sample_rate, _ = parse_wfdb_header(header_path)
    samples = np.loadtxt(str(signal_path), dtype="float32")
    return _embedding_from_signal(samples, sample_rate, max_seconds, model_id, cache_dir)


def load_parkinson_rows(data_path: Path) -> list[tuple[str, dict, bool]]:
    """Map each UCI Parkinson's row to ``(subject, voice_quality_metrics, label)``.

    The dataset ships precomputed MDVP measures, not raw audio, so only the
    voice-quality markers shared with the dysarthria rule can be reconstructed.
    The stored jitter/shimmer columns are fractions (e.g. ``0.00784`` for 0.784%),
    so they are scaled by 100 to match the percentage thresholds of the rule.
    """
    rows: list[tuple[str, dict, bool]] = []
    with data_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            name = (record.get("name") or "").strip()
            metrics = {
                "jitter_local": float(record["MDVP:Jitter(%)"]) * 100.0,
                "shimmer_local": float(record["MDVP:Shimmer"]) * 100.0,
                "hnr": float(record["HNR"]),
            }
            labels = (record.get("status") or "0").strip() == "1"
            rows.append((name.rsplit("_", 1)[0], metrics, labels))
    return rows


def _confusion(labels: list[bool], predictions: list[bool]) -> dict:
    tp = sum(1 for label, pred in zip(labels, predictions) if label and pred)
    fp = sum(1 for label, pred in zip(labels, predictions) if not label and pred)
    tn = sum(1 for label, pred in zip(labels, predictions) if not label and not pred)
    fn = sum(1 for label, pred in zip(labels, predictions) if label and not pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "support": total,
        "positives": tp + fn,
        "negatives": tn + fp,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "sensitivity": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def _f1_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    predictions = scores >= 0.5
    tp = int(np.count_nonzero(labels & predictions))
    fp = int(np.count_nonzero(~labels & predictions))
    fn = int(np.count_nonzero(labels & ~predictions))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _roc_auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return None


def _average_precision_from_scores(
    labels: np.ndarray, scores: np.ndarray
) -> float | None:
    from sklearn.metrics import average_precision_score

    try:
        return float(average_precision_score(labels, scores))
    except ValueError:
        return None


def _bootstrap_interval(
    labels: list[bool],
    scores: list[float],
    metric_fn,
    rounds: int = BOOTSTRAP_ROUNDS,
) -> dict:
    """Percentile bootstrap 95% confidence interval for a metric.

    Resamples recordings with replacement; resamples that contain a single class
    are skipped because the metric is undefined for them.
    """
    labels_array = np.asarray(labels, dtype=bool)
    scores_array = np.asarray(scores, dtype=float)
    total = labels_array.size
    if total == 0:
        return {"lower": None, "upper": None, "resamples": 0}

    rng = np.random.default_rng(RANDOM_STATE)
    values: list[float] = []
    for _ in range(rounds):
        indices = rng.integers(0, total, total)
        sample_labels = labels_array[indices]
        if sample_labels.min() == sample_labels.max():
            continue
        value = metric_fn(sample_labels, scores_array[indices])
        if value is not None and np.isfinite(value):
            values.append(float(value))

    if not values:
        return {"lower": None, "upper": None, "resamples": 0}

    lower = float(np.percentile(values, 100 * 0.025))
    upper = float(np.percentile(values, 100 * 0.975))
    return {"lower": round(lower, 4), "upper": round(upper, 4), "resamples": len(values)}


def _evaluate_rules(
    labels: list[bool],
    codes: list[set[str]],
    evaluated_codes: tuple[str, ...] = WARNING_CODES,
) -> dict:
    results: dict[str, dict] = {}
    for code in evaluated_codes:
        predictions = [code in item for item in codes]
        result = _confusion(labels, predictions)
        result["f1_ci95"] = _bootstrap_interval(
            labels, [1.0 if value else 0.0 for value in predictions], _f1_from_scores
        )
        results[code] = result

    predictions = [bool(item & set(evaluated_codes)) for item in codes]
    result = _confusion(labels, predictions)
    result["f1_ci95"] = _bootstrap_interval(
        labels, [1.0 if value else 0.0 for value in predictions], _f1_from_scores
    )
    results["qualquer_alerta"] = result

    baseline = [_confusion(labels, [False] * len(labels)), _confusion(labels, [True] * len(labels))]
    results["baseline_sempre_normal"] = baseline[0]
    results["baseline_sempre_alerta"] = baseline[1]
    return results


def _build_model(model_name: str = "gradient_boosting"):
    if model_name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        )

    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


MODEL_LABELS = {
    "gradient_boosting": "HistGradientBoostingClassifier",
    "logistic": "StandardScaler + LogisticRegression",
}
PRODUCTION_MODEL = "gradient_boosting"


def _feature_matrix(rows: list[dict], keys: tuple[str, ...] = FEATURE_KEYS) -> np.ndarray:
    return np.asarray(
        [[float(row.get(key, 0.0) or 0.0) for key in keys] for row in rows],
        dtype=np.float64,
    )


def _evaluate_learned(
    features: np.ndarray,
    labels: list[bool],
    groups: list[str],
    feature_names: tuple[str, ...] = FEATURE_KEYS,
    model_name: str = PRODUCTION_MODEL,
) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

    classes = sorted(set(labels))
    if len(classes) < 2:
        return {"status": "skipped", "reason": "proxy label has a single class"}

    fold_counts = [labels.count(value) for value in classes]
    if min(fold_counts) < CV_FOLDS:
        return {
            "status": "skipped",
            "reason": (
                f"smallest class has {min(fold_counts)} samples, fewer than the "
                f"{CV_FOLDS} cross-validation folds"
            ),
        }
    if len(set(groups)) < CV_FOLDS:
        return {
            "status": "skipped",
            "reason": f"only {len(set(groups))} distinct subjects available",
        }

    target = np.asarray([1 if label else 0 for label in labels], dtype=int)
    target_bool = np.asarray(labels, dtype=bool)
    splitter = StratifiedGroupKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    model = _build_model(model_name)

    probabilities = cross_val_predict(
        model,
        features,
        target,
        cv=splitter,
        groups=groups,
        method="predict_proba",
    )[:, 1]
    predicted = (probabilities >= 0.5).astype(int)

    metrics = _confusion(labels, [bool(value) for value in predicted])
    roc_auc = _roc_auc_from_scores(target_bool, probabilities)
    average_precision = _average_precision_from_scores(target_bool, probabilities)
    metrics["roc_auc"] = round(roc_auc, 4) if roc_auc is not None else None
    metrics["average_precision"] = (
        round(average_precision, 4) if average_precision is not None else None
    )
    metrics["roc_auc_ci95"] = _bootstrap_interval(
        labels, list(probabilities), _roc_auc_from_scores
    )
    metrics["average_precision_ci95"] = _bootstrap_interval(
        labels, list(probabilities), _average_precision_from_scores
    )
    metrics["f1_ci95"] = _bootstrap_interval(
        labels, [1.0 if value else 0.0 for value in predicted], _f1_from_scores
    )

    return {
        "status": "ok",
        "model": MODEL_LABELS.get(model_name, model_name),
        "features": list(feature_names),
        "cross_validation": (
            "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) "
            "agrupado por sujeito"
        ),
        **metrics,
    }


def _embedding_learned_block(
    embeddings: np.ndarray | None,
    labels: list[bool],
    subjects: list[str],
    model_id: str,
) -> dict:
    """Linear probe over the deep embeddings, or a skip reason when unavailable."""
    if not embeddings_available():
        return {
            "status": "skipped",
            "reason": "torch/transformers not installed; deep embeddings unavailable",
        }
    if embeddings is None or embeddings.size == 0:
        return {"status": "skipped", "reason": "no embeddings were extracted"}

    learned = _evaluate_learned(
        embeddings,
        labels,
        subjects,
        (EMBEDDING_FEATURE_LABEL,),
        model_name=EMBEDDING_MODEL_NAME,
    )
    return {
        "model_id": model_id,
        "embedding_version": EMBEDDING_VERSION,
        "embedding_dim": EMBEDDING_DIM,
        "feature": EMBEDDING_FEATURE_LABEL,
        **learned,
    }


def _embedding_vs_acoustic(
    acoustic: dict, embedding: dict, dataset_note: str
) -> dict:
    """Frame the handcrafted-feature baseline against the deep-embedding probe."""
    return {
        "handcrafted_features": {
            key: acoustic.get(key)
            for key in ("model", "roc_auc", "average_precision", "f1", "recall")
        },
        "deep_embeddings": {
            key: embedding.get(key)
            for key in (
                "status",
                "feature",
                "model",
                "roc_auc",
                "average_precision",
                "f1",
                "recall",
            )
        },
        "note": dataset_note,
    }


def _build_embedding_block(
    embedding_vectors: list[np.ndarray],
    labels: list[bool],
    subjects: list[str],
    model_id: str,
    acoustic: dict,
    note: str,
) -> dict:
    """Bundle the deep-embedding probe with its comparison against the baseline."""
    matrix = np.vstack(embedding_vectors) if embedding_vectors else None
    probe = _embedding_learned_block(matrix, labels, subjects, model_id)
    return {
        "deep_embeddings_probe": probe,
        "comparison": _embedding_vs_acoustic(acoustic, probe, note),
    }


def _device_summary(devices: list[str], diagnosis_labels: list[bool]) -> dict:
    summary: dict[str, dict] = {}
    for device in sorted(set(devices)):
        indices = [index for index, value in enumerate(devices) if value == device]
        summary[device] = {
            "recordings": len(indices),
            "healthy_recordings": sum(
                1 for index in indices if not diagnosis_labels[index]
            ),
        }
    return summary


def calibrate_voice_quality_limits(healthy_metrics: list[dict]) -> dict:
    """Data-driven normophonia cut-offs derived from the healthy subset.

    Jitter/shimmer limits use a high percentile and the HNR limit a low
    percentile of the healthy distribution, so the rule fires only on values
    that fall outside the observed healthy range.
    """
    if not healthy_metrics:
        return dict(DEFAULT_LIMITS)
    jitter = [float(row.get("jitter_local", 0.0) or 0.0) for row in healthy_metrics]
    shimmer = [float(row.get("shimmer_local", 0.0) or 0.0) for row in healthy_metrics]
    hnr = [float(row.get("hnr", 0.0) or 0.0) for row in healthy_metrics]
    return {
        "jitter_local": round(
            float(np.percentile(jitter, CALIBRATION_HIGH_PERCENTILE)), 4
        ),
        "shimmer_local": round(
            float(np.percentile(shimmer, CALIBRATION_HIGH_PERCENTILE)), 4
        ),
        "hnr": round(float(np.percentile(hnr, CALIBRATION_LOW_PERCENTILE)), 4),
    }


def _calibrated_dysarthria_block(
    metrics_rows: list[dict], labels: list[bool]
) -> dict:
    """Re-evaluate the dysarthria rule with thresholds calibrated on the healthy rows.

    Reports how the screening rule behaves when its normophonia cut-offs come from
    the observed healthy distribution instead of fixed Praat/MDVP norms. The
    calibration reuses the same dataset's healthy subset, so it illustrates the
    effect of threshold choice rather than claiming an independent validation.
    """
    healthy = [row for row, label in zip(metrics_rows, labels) if not label]
    limits = calibrate_voice_quality_limits(healthy)
    codes = [
        {item["code"] for item in dysarthria_findings(row, limits)}
        for row in metrics_rows
    ]
    return {
        "limits": limits,
        "default_limits": dict(DEFAULT_LIMITS),
        "rule_based": _evaluate_rules(labels, codes, DYSARTHRIA_CODES),
    }


def _warning_codes(row: dict) -> set[str]:
    """Warning codes fired by a metric row (shared by every dataset)."""
    return {
        finding["code"]
        for finding in acoustic_findings(row)
        if finding["code"] in WARNING_CODES
    }


def collect_voiced_records(
    voiced_root: Path, target_sample_rate: int, max_seconds: float, limit: int | None
) -> list[dict]:
    """Load every VOICED recording into a shared per-recording record."""
    recordings = iter_voiced_recordings(voiced_root)
    if limit:
        recordings = recordings[:limit]
    records: list[dict] = []
    for signal_path, header_path, diagnosis in recordings:
        row = load_voiced_metrics(
            signal_path, header_path, target_sample_rate, max_seconds
        )
        records.append(
            {
                "signal_path": signal_path,
                "header_path": header_path,
                "diagnosis": diagnosis,
                "metrics": row,
                "codes": _warning_codes(row),
                "label": diagnosis.lower() != "healthy",
                "subject": signal_path.stem,
            }
        )
    return records


def collect_icbhi_records(
    data_root: Path, target_sample_rate: int, max_seconds: float, limit: int | None
) -> list[dict]:
    """Load every ICBHI recording into a shared per-recording record."""
    diagnosis_path = data_root / "ICBHI_Challenge_diagnosis.txt"
    diagnosis = load_diagnosis(diagnosis_path) if diagnosis_path.is_file() else {}
    pairs = iter_recordings(data_root)
    if limit:
        pairs = pairs[:limit]
    records: list[dict] = []
    for wav_path, annotation_path in pairs:
        cycles = parse_cycle_annotations(annotation_path)
        if not cycles:
            continue
        row = load_metrics(wav_path, target_sample_rate, max_seconds)
        patient_id = wav_path.name.split("_")[0]
        subject_diagnosis = diagnosis.get(patient_id)
        records.append(
            {
                "wav_path": wav_path,
                "metrics": row,
                "codes": _warning_codes(row),
                "cycle_label": any(crackle or wheeze for crackle, wheeze in cycles),
                "diagnosis_label": (
                    subject_diagnosis is not None and subject_diagnosis != "Healthy"
                ),
                "missing_diagnosis": subject_diagnosis is None,
                "subject": patient_id,
                "device": wav_path.stem.split("_")[-1],
            }
        )
    return records


def _cross_transfer(
    source_features: np.ndarray,
    source_labels: list[bool],
    target_features: np.ndarray,
    target_labels: list[bool],
) -> dict:
    """Train on one dataset and evaluate on another (domain-shift check)."""
    target = np.asarray([1 if label else 0 for label in source_labels], dtype=int)
    model = _build_model(PRODUCTION_MODEL)
    model.fit(source_features, target)
    probabilities = model.predict_proba(target_features)[:, 1]
    metrics = _confusion(target_labels, [bool(value >= 0.5) for value in probabilities])
    target_bool = np.asarray(target_labels, dtype=bool)
    metrics["roc_auc"] = _roc_auc_from_scores(target_bool, probabilities)
    metrics["average_precision"] = _average_precision_from_scores(
        target_bool, probabilities
    )
    return metrics


def evaluate_cross(
    voiced_root: Path,
    icbhi_root: Path,
    target_sample_rate: int,
    max_seconds: float,
    limit: int | None,
) -> dict:
    voiced_features, voiced_labels, _ = collect_voiced_records(
        voiced_root, target_sample_rate, max_seconds, limit
    )
    icbhi_features, icbhi_labels, _ = collect_icbhi_records(
        icbhi_root, target_sample_rate, max_seconds, limit
    )
    return {
        "dataset": {
            "name": "VOICED x ICBHI cross-dataset transfer",
            "source": "VOICED (PhysioNet) + ICBHI 2017",
            "voiced_recordings": len(voiced_labels),
            "icbhi_recordings": len(icbhi_labels),
            "preprocessing": {
                "target_sample_rate_hz": target_sample_rate,
                "max_seconds": max_seconds,
            },
        },
        "voiced_to_icbhi": _cross_transfer(
            voiced_features, voiced_labels, icbhi_features, icbhi_labels
        ),
        "icbhi_to_voiced": _cross_transfer(
            icbhi_features, icbhi_labels, voiced_features, voiced_labels
        ),
        "note": (
            "Both directions transfer a model trained on one domain to a different "
            "one (sustained vowel vs respiratory recording). ROC-AUC near 0.5 means "
            "the acoustic layer does not generalise across domains, which frames the "
            "proxy as weak rather than claiming clinical detection."
        ),
    }


def evaluate(
    data_root: Path,
    target_sample_rate: int,
    max_seconds: float,
    limit: int | None,
    include_embeddings: bool = False,
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    embedding_cache_dir: str | None = None,
) -> dict:
    records = collect_icbhi_records(data_root, target_sample_rate, max_seconds, limit)
    if not records:
        raise RuntimeError(f"No .wav/.txt pairs found under {data_root.resolve()}")

    metrics_rows = [record["metrics"] for record in records]
    cycle_labels = [record["cycle_label"] for record in records]
    diagnosis_labels = [record["diagnosis_label"] for record in records]
    prediction_codes = [record["codes"] for record in records]
    subject_ids = [record["subject"] for record in records]
    devices = [record["device"] for record in records]
    abstained = sum(1 for record in records if not record["codes"])
    missing_diagnosis = sum(1 for record in records if record["missing_diagnosis"])

    embedding_vectors: list[np.ndarray] = []
    if include_embeddings:
        for record in records:
            vector = load_embedding(
                record["wav_path"], max_seconds, embedding_model_id, embedding_cache_dir
            )
            if vector is None:
                embedding_vectors = []
                break
            embedding_vectors.append(vector)

    features = _feature_matrix(metrics_rows)
    learned_primary = _evaluate_learned(features, cycle_labels, subject_ids)
    learned_secondary = _evaluate_learned(features, diagnosis_labels, subject_ids)

    embedding_block: dict | None = None
    if include_embeddings:
        embedding_block = _build_embedding_block(
            embedding_vectors,
            cycle_labels,
            subject_ids,
            embedding_model_id,
            learned_primary,
            "ICBHI lung sounds, not speech: the experiment checks whether a "
            "self-supervised speech/language embedding transfers to respiratory "
            "audio at all, against the handcrafted features.",
        )

    total_abnormal = sum(1 for label in cycle_labels if label)

    report = {
        "dataset": {
            "name": "ICBHI 2017 Respiratory Sound Database",
            "source": "https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge",
            "citation": DATASET_CITATION,
            "license": DATASET_LICENSE,
            "recordings_used": len(cycle_labels),
            "recordings_abnormal": total_abnormal,
            "recordings_normal": len(cycle_labels) - total_abnormal,
            "subjects_without_diagnosis": missing_diagnosis,
            "preprocessing": {
                "target_sample_rate_hz": target_sample_rate,
                "max_seconds": max_seconds,
                "resampling": "box-average anti-alias + decimation (numpy)",
            },
        },
        "proxy_label_primary": {
            "name": "crackle_or_wheeze_present",
            "definition": (
                "Recording is abnormal when at least one annotated respiratory "
                "cycle contains crackles or wheezes."
            ),
            "rule_based": _evaluate_rules(cycle_labels, prediction_codes),
            "learned_baseline": learned_primary,
            "deep_embeddings": embedding_block,
        },
        "proxy_label_secondary": {
            "name": "subject_diagnosis_not_healthy",
            "definition": (
                "Recording is abnormal when the published subject diagnosis is "
                "anything other than Healthy."
            ),
            "rule_based": _evaluate_rules(diagnosis_labels, prediction_codes),
            "learned_baseline": learned_secondary,
        },
        "abstentions": {
            "recordings_without_warning": abstained,
            "note": (
                "No warning code fired for these recordings; they count as "
                "negative predictions in the tables above."
            ),
        },
        "confounds": {
            "recording_device_vs_healthy_label": _device_summary(
                devices, diagnosis_labels
            ),
            "note": (
                "Every Healthy subject in ICBHI was recorded with a single device "
                "and the four devices differ sharply in signal level. The features "
                "therefore encode the recording equipment, which is why the "
                "diagnosis-based label is predicted far better than the crackle/"
                "wheeze label. That secondary result is device recognition and must "
                "not be read as clinical capability."
            ),
        },
        "limitations": [
            "ICBHI contains lung sounds, not physiotherapy session speech.",
            "The acoustic flags were tuned for speech; this experiment tests whether "
            "they retain any signal on respiratory audio.",
            "No public dataset labels fatigue or dyspnoea during rehabilitation, so "
            "the fatigue flag itself cannot be validated here.",
            "Crackles/wheezes are a proxy for respiratory abnormality, not a clinical "
            "ground truth for the flags.",
            "The learned baseline uses subject-grouped cross-validation; without "
            "grouping, several recordings from the same subject leak across folds and "
            "inflate the metrics.",
        ],
    }
    return report


def evaluate_voiced(
    voiced_root: Path,
    target_sample_rate: int,
    max_seconds: float,
    limit: int | None,
    model_path: Path | None,
    include_embeddings: bool = False,
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    embedding_cache_dir: str | None = None,
) -> dict:
    records = collect_voiced_records(voiced_root, target_sample_rate, max_seconds, limit)
    if not records:
        raise RuntimeError(f"No VOICED recordings found under {voiced_root.resolve()}")

    metrics_rows = [record["metrics"] for record in records]
    labels = [record["label"] for record in records]
    prediction_codes = [record["codes"] for record in records]
    subject_ids = [record["subject"] for record in records]
    diagnoses: Counter = Counter(record["diagnosis"].lower() for record in records)
    abstained = sum(1 for record in records if not record["codes"])

    embedding_vectors: list[np.ndarray] = []
    if include_embeddings:
        for record in records:
            vector = load_voiced_embedding(
                record["signal_path"],
                record["header_path"],
                max_seconds,
                embedding_model_id,
                embedding_cache_dir,
            )
            if vector is None:
                embedding_vectors = []
                break
            embedding_vectors.append(vector)

    features = _feature_matrix(metrics_rows)
    positives = sum(1 for label in labels if label)
    rule_based = _evaluate_rules(labels, prediction_codes)
    learned = _evaluate_learned(features, labels, subject_ids)

    embedding_block: dict | None = None
    if include_embeddings:
        embedding_block = _build_embedding_block(
            embedding_vectors,
            labels,
            subject_ids,
            embedding_model_id,
            learned,
            "VOICED sustained-vowel voice: the experiment checks whether a "
            "self-supervised speech embedding beats the handcrafted acoustic "
            "features on the pathological-voice proxy label.",
        )

    artifact_info = None
    if model_path is not None:
        artifact_info = train_voice_model(features, labels, learned, model_path)

    report = {
        "dataset": {
            "name": "VOICED Database (VOice ICar fEDerico II)",
            "source": "https://physionet.org/content/voiced/1.0.0/",
            "citation": VOICED_DATASET_CITATION,
            "license": VOICED_DATASET_LICENSE,
            "recordings_used": len(labels),
            "recordings_pathological": positives,
            "recordings_healthy": len(labels) - positives,
            "diagnosis_counts": dict(sorted(diagnoses.items())),
            "preprocessing": {
                "target_sample_rate_hz": target_sample_rate,
                "max_seconds": max_seconds,
                "resampling": "box-average anti-alias + decimation (numpy)",
            },
        },
        "proxy_label_primary": {
            "name": "pathological_voice",
            "definition": (
                "Recording is positive when the clinically verified diagnosis is "
                "anything other than healthy."
            ),
            "rule_based": rule_based,
            "learned_baseline": learned,
            "deep_embeddings": embedding_block,
            "dysarthria_calibrated": _calibrated_dysarthria_block(
                metrics_rows, labels
            ),
        },
        "artifact": artifact_info,
        "abstentions": {
            "recordings_without_warning": abstained,
            "note": (
                "No warning code fired for these recordings; they count as "
                "negative predictions in the tables above."
            ),
        },
        "limitations": [
            "VOICED contains sustained vowel phonation, not physiotherapy session "
            "speech; it is a proxy for the vocal-alteration claim.",
            "A sustained vowel has no pauses or speech onsets, so the pause-based "
            "flags (possivel_fadiga, possivel_desconforto_respiratorio) cannot fire "
            "and are not exercised here.",
            "No public dataset labels fatigue or dyspnoea during rehabilitation, so "
            "the rehabilitation-specific use case remains unvalidated.",
            "The learned baseline uses subject-grouped cross-validation; each subject "
            "contributes a single recording.",
            "The deep-embedding probe is a self-supervised model pretrained on "
            "spontaneous/read speech and applied to a sustained vowel; it is a "
            "comparison of representations, not a clinically validated classifier.",
        ],
    }
    return report


def evaluate_parkinson(data_path: Path, limit: int | None) -> dict:
    rows = load_parkinson_rows(data_path)
    if limit:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"No rows found in {data_path.resolve()}")

    metrics_rows = [metrics for _, metrics, _ in rows]
    labels = [label for _, _, label in rows]
    subject_ids = [subject for subject, _, _ in rows]

    prediction_codes: list[set[str]] = []
    abstained = 0
    for _, metrics, _ in rows:
        codes = {item["code"] for item in dysarthria_findings(metrics)}
        if not codes:
            abstained += 1
        prediction_codes.append(codes)

    features = _feature_matrix(metrics_rows, DYSARTHRIA_FEATURE_KEYS)
    positives = sum(1 for label in labels if label)

    return {
        "dataset": {
            "name": "UCI Parkinson's Disease (sustained-vowel voice measures)",
            "source": (
                "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/"
            ),
            "citation": PARKINSON_DATASET_CITATION,
            "license": PARKINSON_DATASET_LICENSE,
            "recordings_used": len(labels),
            "recordings_pathological": positives,
            "recordings_healthy": len(labels) - positives,
            "subjects": len(set(subject_ids)),
        },
        "proxy_label_primary": {
            "name": "parkinson_diagnosis",
            "definition": (
                "Row is positive when the published label indicates Parkinson's "
                "disease (status = 1)."
            ),
            "rule_based": _evaluate_rules(labels, prediction_codes, DYSARTHRIA_CODES),
            "learned_baseline": _evaluate_learned(
                # Only three voice-quality markers here; a linear model suits that
                # small feature set better than the boosted trees used elsewhere.
                features,
                labels,
                subject_ids,
                DYSARTHRIA_FEATURE_KEYS,
                model_name="logistic",
            ),
            "dysarthria_calibrated": _calibrated_dysarthria_block(
                metrics_rows, labels
            ),
        },
        "abstentions": {
            "recordings_without_warning": abstained,
            "note": (
                "No dysarthria marker fired for these recordings; they count as "
                "negative predictions in the tables above."
            ),
        },
        "limitations": [
            "The dataset ships precomputed MDVP voice measures, not raw audio; only "
            "jitter/shimmer/HNR can be reconstructed, so the full session pipeline is "
            "not exercised here.",
            "Parkinson's hypokinetic dysarthria is a proxy for the dysarthria flag, "
            "not a physiotherapy-session ground truth.",
            "The learned baseline uses subject-grouped cross-validation; several "
            "sustained-vowel recordings come from the same speaker.",
        ],
    }


def train_voice_model(
    features: np.ndarray,
    labels: list[bool],
    validation: dict,
    model_path: Path,
) -> dict:
    """Fit the reusable voice-alteration classifier and persist it with its metrics."""
    target = np.asarray([1 if label else 0 for label in labels], dtype=int)
    model = _build_model(PRODUCTION_MODEL)
    model.fit(features, target)

    artifact = {
        "model": model,
        "model_name": MODEL_LABELS.get(PRODUCTION_MODEL, PRODUCTION_MODEL),
        "feature_version": FEATURE_VERSION,
        "features": list(FEATURE_KEYS),
        "dataset": "VOICED",
        "label_names": {0: "healthy", 1: "pathological"},
        "validation": validation,
        "limitations": (
            "Sustained-vowel proxy (VOICED). Not validated on physiotherapy "
            "session speech; treat as experimental support only."
        ),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return {
        "path": str(model_path.relative_to(PROJECT_ROOT)),
        "feature_version": FEATURE_VERSION,
        "validation_roc_auc": validation.get("roc_auc"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("icbhi", "voiced", "parkinson", "cross"),
        default="icbhi",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--voiced-root", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--target-sample-rate", type=int, default=DEFAULT_TARGET_SAMPLE_RATE
    )
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument(
        "--limit", type=int, default=None, help="evaluate only the first N recordings"
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help=(
            "also train a linear probe on wav2vec2 deep embeddings and compare it "
            "with the handcrafted features (needs torch/transformers; icbhi/voiced only)"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL_ID,
        help="HuggingFace model id used for the deep embeddings",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="optional HuggingFace cache directory for the embedding model",
    )
    args = parser.parse_args()

    embedding_cache = str(args.embedding_cache) if args.embedding_cache else None

    if args.dataset == "icbhi":
        data_root = args.data_root or DATA_ROOT
        report_path = args.report_path or REPORT_PATH
        report = evaluate(
            data_root=data_root,
            target_sample_rate=args.target_sample_rate,
            max_seconds=args.max_seconds,
            limit=args.limit,
            include_embeddings=args.include_embeddings,
            embedding_model_id=args.embedding_model,
            embedding_cache_dir=embedding_cache,
        )
    elif args.dataset == "voiced":
        data_root = args.data_root or VOICED_DATA_ROOT
        report_path = args.report_path or VOICED_REPORT_PATH
        report = evaluate_voiced(
            voiced_root=data_root,
            target_sample_rate=args.target_sample_rate,
            max_seconds=args.max_seconds,
            limit=args.limit,
            model_path=args.model_path or VOICED_MODEL_PATH,
            include_embeddings=args.include_embeddings,
            embedding_model_id=args.embedding_model,
            embedding_cache_dir=embedding_cache,
        )
    elif args.dataset == "cross":
        report_path = args.report_path or CROSS_REPORT_PATH
        report = evaluate_cross(
            voiced_root=args.voiced_root or VOICED_DATA_ROOT,
            icbhi_root=args.data_root or DATA_ROOT,
            target_sample_rate=args.target_sample_rate,
            max_seconds=args.max_seconds,
            limit=args.limit,
        )
    else:
        data_path = args.data_root or PARKINSON_DATA_PATH
        report_path = args.report_path or PARKINSON_REPORT_PATH
        report = evaluate_parkinson(data_path=data_path, limit=args.limit)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    _print_summary(report, report_path)


def _print_summary(report: dict, report_path: Path) -> None:
    dataset = report["dataset"]
    if "recordings_used" in dataset:
        print(f"{dataset['name']}: {dataset['recordings_used']} recordings")
    else:
        print(
            f"{dataset['name']}: "
            f"voiced={dataset.get('voiced_recordings')} "
            f"icbhi={dataset.get('icbhi_recordings')}"
        )

    blocks = [
        key
        for key in ("proxy_label_primary", "proxy_label_secondary")
        if key in report
    ]
    for label in blocks:
        block = report[label]
        rules = block["rule_based"]["qualquer_alerta"]
        learned = block["learned_baseline"]
        print(f"\n{label} ({block['name']})")
        print(
            f"  qualquer_alerta  f1={rules['f1']:.3f} "
            f"precision={rules['precision']:.3f} recall={rules['recall']:.3f} "
            f"specificity={rules['specificity']:.3f} accuracy={rules['accuracy']:.3f}"
        )
        if learned.get("status") == "ok":
            print(
                f"  {learned['model']}  f1={learned['f1']:.3f} "
                f"precision={learned['precision']:.3f} "
                f"recall={learned['recall']:.3f} "
                f"specificity={learned['specificity']:.3f} "
                f"roc_auc={learned['roc_auc']} "
                f"ap={learned['average_precision']}"
            )
        calibrated = block.get("dysarthria_calibrated")
        if calibrated:
            calib_rules = calibrated["rule_based"]["qualquer_alerta"]
            print(
                f"  disartria calibrada  f1={calib_rules['f1']:.3f} "
                f"recall={calib_rules['recall']:.3f} "
                f"specificity={calib_rules['specificity']:.3f} "
                f"limites={calibrated['limits']}"
            )
        embeddings = block.get("deep_embeddings")
        if embeddings:
            probe = embeddings["deep_embeddings_probe"]
            if probe.get("status") == "ok":
                print(
                    f"  deep embeddings ({probe['embedding_dim']}d)  "
                    f"f1={probe['f1']:.3f} recall={probe['recall']:.3f} "
                    f"specificity={probe['specificity']:.3f} "
                    f"roc_auc={probe['roc_auc']} ap={probe['average_precision']}"
                )
            else:
                print(f"  deep embeddings: skipped ({probe.get('reason')})")

    for key in ("voiced_to_icbhi", "icbhi_to_voiced"):
        if key in report:
            transfer = report[key]
            print(
                f"\n{key}  f1={transfer['f1']:.3f} "
                f"recall={transfer['recall']:.3f} "
                f"specificity={transfer['specificity']:.3f} "
                f"roc_auc={transfer['roc_auc']} "
                f"ap={transfer['average_precision']}"
            )

    artifact = report.get("artifact")
    if artifact:
        print(
            f"\nartifact: {artifact['path']} "
            f"(validation roc_auc={artifact['validation_roc_auc']})"
        )
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
