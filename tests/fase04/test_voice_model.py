from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.rehab_audio.acoustics import FEATURE_KEYS, FEATURE_VERSION
from backend.rehab_audio.voice_model import predict_voice_alteration


def _fit_model() -> object:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, len(FEATURE_KEYS)))
    labels = np.array([0, 1] * 30)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    model.fit(features, labels)
    return model


def _write_artifact(tmp_path, *, feature_version: str, model=NotImplemented):
    artifact = {
        "model": _fit_model() if model is NotImplemented else model,
        "model_name": "StandaloneTestModel",
        "feature_version": feature_version,
        "features": list(FEATURE_KEYS),
        "dataset": "VOICED",
        "label_names": {0: "healthy", 1: "pathological"},
        "validation": {"roc_auc": 0.6},
    }
    path = tmp_path / "voice_alteration_classifier.joblib"
    joblib.dump(artifact, path)
    return path


def test_predict_returns_none_when_artifact_missing(tmp_path):
    assert predict_voice_alteration({}, model_path=tmp_path / "absent.joblib") is None


def test_predict_returns_probability_and_flag(tmp_path):
    path = _write_artifact(tmp_path, feature_version=FEATURE_VERSION)
    metrics = {key: 0.1 for key in FEATURE_KEYS}

    result = predict_voice_alteration(metrics, model_path=path)

    assert result is not None
    assert 0.0 <= result["probability_pathological_voice"] <= 1.0
    assert result["threshold"] == 0.5
    assert result["flagged"] is (result["probability_pathological_voice"] >= 0.5)
    assert result["dataset"] == "VOICED"
    assert result["model"] == "StandaloneTestModel"
    assert result["feature_version"] == FEATURE_VERSION


def test_predict_raises_on_feature_version_mismatch(tmp_path):
    path = _write_artifact(tmp_path, feature_version="rehab-audio-acoustic-v0")

    with pytest.raises(RuntimeError):
        predict_voice_alteration({key: 0.1 for key in FEATURE_KEYS}, model_path=path)
