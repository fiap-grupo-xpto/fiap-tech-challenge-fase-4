from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "entregas" / "fase04" / "src"))

from backend.rehab_audio.embeddings import (  # noqa: E402
    EMBEDDING_DIM,
    EMBEDDING_SAMPLE_RATE,
    embeddings_available,
    extract_embeddings,
    resample_signal,
)

import evaluate_audio_flags as harness  # noqa: E402


def test_resample_upsamples_to_target_length():
    signal = np.sin(np.linspace(0, 2 * np.pi * 100, 8000)).astype(np.float32)

    resampled = resample_signal(signal, 8000, 16000)

    assert resampled.size == 16000
    assert np.all(np.isfinite(resampled))


def test_resample_is_noop_when_already_at_target_rate():
    signal = np.ones(100, dtype=np.float32)

    assert resample_signal(signal, 16000, 16000) is signal


def test_resample_handles_empty_signal():
    signal = np.asarray([], dtype=np.float32)

    assert resample_signal(signal, 8000, 16000).size == 0


def test_embeddings_available_returns_bool():
    assert isinstance(embeddings_available(), bool)


def test_embedding_learned_block_skips_without_matrix():
    result = harness._embedding_learned_block(
        None, [True, False], ["a", "b"], harness.EMBEDDING_MODEL_ID
    )

    assert result["status"] == "skipped"
    assert "reason" in result


def test_embedding_vs_acoustic_frames_both_representations():
    acoustic = {"model": "HistGradientBoostingClassifier", "roc_auc": 0.61, "f1": 0.5}
    embedding = {
        "status": "ok",
        "feature": harness.EMBEDDING_FEATURE_LABEL,
        "model": "StandardScaler + LogisticRegression",
        "roc_auc": 0.7,
        "f1": 0.6,
    }

    comparison = harness._embedding_vs_acoustic(acoustic, embedding, "note")

    assert comparison["handcrafted_features"]["roc_auc"] == 0.61
    assert comparison["deep_embeddings"]["roc_auc"] == 0.7
    assert comparison["note"] == "note"


@pytest.mark.skipif(
    not embeddings_available(), reason="torch/transformers not installed"
)
def test_extract_embeddings_returns_fixed_dimension_vector():
    sample_rate = EMBEDDING_SAMPLE_RATE
    times = np.arange(sample_rate) / sample_rate
    signal = 0.3 * np.sin(2 * np.pi * 220 * times)

    vector = extract_embeddings(signal, sample_rate)

    assert vector is not None
    assert vector.shape == (EMBEDDING_DIM,)
    assert np.all(np.isfinite(vector))
