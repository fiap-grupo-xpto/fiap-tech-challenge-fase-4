"""Pretrained deep audio embeddings for the rehabilitation audio experiment.

Optional layer that mirrors the acoustic pipeline: a self-supervised speech
model (wav2vec2) turns the raw waveform into a mean-pooled embedding, so the
offline harness (``entregas/fase04/src/evaluate_audio_flags.py``) can test
whether a learned representation carries more signal than the handcrafted
features on the labeled voice dataset.

The ``torch``/``transformers`` dependency is resolved lazily. When they are
missing, ``embeddings_available()`` is ``False`` and ``extract_embeddings``
returns ``None``, so the core analysis keeps working with the heuristic features
only - the same graceful-degradation rule used by ``voice_model.py`` for the
persisted classifier.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from backend.rehab_audio.acoustics import to_mono


EMBEDDING_MODEL_ID = "facebook/wav2vec2-base"
EMBEDDING_VERSION = "rehab-audio-wav2vec2-v1"
EMBEDDING_SAMPLE_RATE = 16000
EMBEDDING_DIM = 768


def _dependencies_available() -> bool:
    """Return ``True`` only when both ``torch`` and ``transformers`` import."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def embeddings_available() -> bool:
    """Whether the deep-embedding layer can run in this environment."""
    return _dependencies_available()


def resample_signal(
    signal: np.ndarray, sample_rate: int, target_rate: int
) -> np.ndarray:
    """Linear-interpolation resampler (numpy only), used to reach 16 kHz.

    The harness decimates to 8 kHz for the handcrafted features, but wav2vec2
    expects 16 kHz, so embeddings need to upsample instead of downsampling.
    """
    if signal.size == 0 or sample_rate <= 0 or target_rate <= 0:
        return signal
    if sample_rate == target_rate:
        return signal
    target_count = int(round(signal.size / float(sample_rate) * float(target_rate)))
    if target_count <= 1:
        return signal
    source_positions = np.linspace(0.0, signal.size - 1, target_count)
    resampled = np.interp(source_positions, np.arange(signal.size), signal)
    return np.ascontiguousarray(resampled, dtype=np.float32)


@lru_cache(maxsize=2)
def _load_model(model_id: str, cache_dir: str | None = None):
    """Load the feature extractor and model once and keep them in eval mode."""
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
    model.eval()
    return extractor, model


def extract_embeddings(
    samples: np.ndarray,
    sample_rate: int,
    model_id: str = EMBEDDING_MODEL_ID,
    cache_dir: str | None = None,
) -> np.ndarray | None:
    """Return a mean-pooled wav2vec2 embedding, or ``None`` when unavailable."""
    if not embeddings_available():
        return None

    import torch

    signal = to_mono(samples)
    if signal.size == 0 or sample_rate <= 0:
        return None
    signal = resample_signal(signal, int(sample_rate), EMBEDDING_SAMPLE_RATE)

    extractor, model = _load_model(model_id, cache_dir)
    inputs = extractor(
        signal, sampling_rate=EMBEDDING_SAMPLE_RATE, return_tensors="pt"
    )
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state
    pooled = hidden.mean(dim=1).squeeze(0).cpu().numpy()
    return np.ascontiguousarray(pooled, dtype=np.float32)
