"""Reusable voice-alteration classifier consumed by the audio analysis service.

The artifact is produced offline by
``entregas/fase04/src/evaluate_audio_flags.py --dataset voiced`` and mirrors the
video pipeline (``backend/rehab_video/service.py``): a persisted scikit-learn
model is loaded once and applied to the acoustic features of the current audio.

The model is an optional enhancement for the heuristic report. When no artifact
is present the analysis still runs and only the heuristic flags are returned.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from backend.rehab_audio.acoustics import FEATURE_KEYS, FEATURE_VERSION


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "voice_alteration_classifier.joblib"
)
DECISION_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _load_artifact(model_path: str) -> dict:
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise RuntimeError("The voice-alteration artifact is not a valid mapping.")
    if artifact.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError(
            "The voice-alteration model uses an incompatible feature definition. "
            "Retrain it with "
            "`python entregas/fase04/src/evaluate_audio_flags.py --dataset voiced`."
        )
    return artifact


def predict_voice_alteration(
    metrics: dict,
    model_path: Path | str = MODEL_PATH,
) -> dict | None:
    """Return the voice-alteration estimate, or ``None`` when no model exists."""
    artifact_path = Path(model_path)
    if not artifact_path.is_file():
        return None

    artifact = _load_artifact(str(artifact_path))
    model = artifact.get("model")
    if model is None:
        raise RuntimeError("The voice-alteration artifact is missing its model.")

    vector = np.asarray(
        [[float(metrics.get(key, 0.0) or 0.0) for key in FEATURE_KEYS]],
        dtype=np.float64,
    )
    probabilities = model.predict_proba(vector)[0]
    class_ids = list(model.classes_)
    probability = float(probabilities[class_ids.index(1)])
    validation = artifact.get("validation") or {}

    return {
        "probability_pathological_voice": round(probability, 4),
        "threshold": DECISION_THRESHOLD,
        "flagged": bool(probability >= DECISION_THRESHOLD),
        "model": str(artifact.get("model_name") or type(model).__name__),
        "dataset": str(artifact.get("dataset", "")),
        "validation_roc_auc": validation.get("roc_auc"),
        "feature_version": str(artifact.get("feature_version", "")),
    }
