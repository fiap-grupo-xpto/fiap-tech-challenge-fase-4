from __future__ import annotations

import math

import numpy as np
import parselmouth

from backend.rehab_audio.dysarthria import (
    DYSARTHRIA_FEATURE_KEYS,
    dysarthria_findings,
)


FEATURE_VERSION = "rehab-audio-acoustic-v2"
# Spectral/cepstral features computed from the voiced frames of the signal.
SPECTRAL_FEATURE_KEYS = (
    "spectral_centroid_hz",
    "spectral_rolloff_hz",
    "spectral_flux",
    "zero_crossing_rate",
)
# Core features consumed by the persisted voice-alteration classifier.
FEATURE_KEYS = (
    "duration_seconds",
    "rms_energy",
    "voiced_ratio",
    "pause_ratio",
    "mean_pause_seconds",
    "speaking_rate",
    "pitch_variability",
    *SPECTRAL_FEATURE_KEYS,
)
# Full metric set exposed in the report (core features + voice-quality markers).
METRIC_KEYS = FEATURE_KEYS + DYSARTHRIA_FEATURE_KEYS

# Sample rate the features are computed at. Training (entregas/fase04/src/
# evaluate_audio_flags.py) and inference (service.py) must agree on it, otherwise
# the absolute-Hz spectral features drift between the two.
TARGET_FEATURE_SAMPLE_RATE = 8000

MIN_DURATION_SECONDS = 1.0
MIN_RMS = 1e-4
PAUSE_RATIO_LIMIT = 0.45
LOW_ENERGY_LIMIT = 0.05
MEAN_PAUSE_LIMIT = 0.8
PAUSE_DENSITY_LIMIT = 0.35
PITCH_VARIABILITY_LIMIT = 0.35

_FRAME_SECONDS = 0.025
_HOP_SECONDS = 0.010
_MIN_PAUSE_SECONDS = 0.2
_F0_MIN_HZ = 70.0
_F0_MAX_HZ = 400.0
_PRAAT_F0_MIN_HZ = 75.0
_PRAAT_F0_MAX_HZ = 600.0
_SPECTRAL_ROLLOFF = 0.85


def _zero_metrics() -> dict:
    return {key: 0.0 for key in METRIC_KEYS}


def to_mono(samples: np.ndarray) -> np.ndarray:
    signal = np.asarray(samples, dtype=np.float32)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    return np.ascontiguousarray(signal, dtype=np.float32).reshape(-1)


def decimate(
    signal: np.ndarray, sample_rate: int, target_rate: int
) -> tuple[np.ndarray, int]:
    """Box-average anti-alias filter followed by decimation (numpy only)."""
    if sample_rate <= target_rate:
        return signal, sample_rate
    factor = max(1, int(round(sample_rate / float(target_rate))))
    if factor <= 1:
        return signal, sample_rate
    usable = (signal.size // factor) * factor
    if usable == 0:
        return signal, sample_rate
    reduced = signal[:usable].reshape(-1, factor).mean(axis=1)
    return np.ascontiguousarray(reduced, dtype=np.float32), int(round(sample_rate / factor))


def _frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if signal.size < frame_length:
        return signal.reshape(1, -1)
    frame_count = 1 + (signal.size - frame_length) // hop_length
    return np.lib.stride_tricks.as_strided(
        signal,
        shape=(frame_count, frame_length),
        strides=(signal.strides[0] * hop_length, signal.strides[0]),
        writeable=False,
    )


def _run_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in mask:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _estimate_f0(frame: np.ndarray, sample_rate: int) -> float:
    centered = frame - float(np.mean(frame))
    if float(np.sqrt(np.mean(np.square(centered)))) < MIN_RMS:
        return float("nan")
    correlation = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
    if correlation.size == 0 or correlation[0] <= 0:
        return float("nan")
    correlation = correlation / correlation[0]
    min_lag = max(1, int(sample_rate / _F0_MAX_HZ))
    max_lag = min(centered.size - 1, int(sample_rate / _F0_MIN_HZ))
    if max_lag <= min_lag:
        return float("nan")
    window = correlation[min_lag : max_lag + 1]
    peak = int(np.argmax(window))
    if window[peak] <= 0:
        return float("nan")
    lag = min_lag + peak
    if lag <= 0:
        return float("nan")
    return float(sample_rate) / float(lag)


def _finite(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _voice_quality_metrics(signal: np.ndarray, sample_rate: int) -> dict:
    """Praat-based jitter/shimmer/HNR over the voiced part (0.0 when unknown)."""
    zero = {key: 0.0 for key in DYSARTHRIA_FEATURE_KEYS}
    if signal.size == 0 or sample_rate <= 0:
        return zero
    if float(np.sqrt(np.mean(np.square(signal)))) < MIN_RMS:
        return zero
    try:
        sound = parselmouth.Sound(
            signal.astype(np.float64), sampling_frequency=float(sample_rate)
        )
        point_process = parselmouth.praat.call(
            sound,
            "To PointProcess (periodic, cc)",
            _PRAAT_F0_MIN_HZ,
            _PRAAT_F0_MAX_HZ,
        )
        jitter = parselmouth.praat.call(
            point_process, "Get jitter (local)", 0.0, 0.0, 1e-4, 0.02, 1.3
        )
        shimmer = parselmouth.praat.call(
            [sound, point_process], "Get shimmer (local)", 0.0, 0.0, 1e-4, 0.02, 1.3, 1.6
        )
        hnr = parselmouth.praat.call(sound.to_harmonicity(), "Get mean", 0.0, 0.0)
    except Exception:  # noqa: BLE001 - Praat raises opaque errors on odd signals
        return zero
    return {
        "jitter_local": round(_finite(jitter) * 100.0, 4),
        "shimmer_local": round(_finite(shimmer) * 100.0, 4),
        "hnr": round(_finite(hnr), 4),
    }


def _spectral_metrics(
    frames: np.ndarray, sample_rate: int, frame_length: int, voiced: np.ndarray
) -> dict:
    """Spectral/cepstral-shape features over the voiced frames (0.0 when unknown)."""
    zero = {key: 0.0 for key in SPECTRAL_FEATURE_KEYS}
    indices = np.flatnonzero(voiced)
    if indices.size == 0 or frame_length < 2 or sample_rate <= 0:
        return zero

    voiced_frames = frames[indices].astype(np.float64)
    windowed = voiced_frames * np.hanning(frame_length)
    magnitude = np.abs(np.fft.rfft(windowed, axis=1))
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / float(sample_rate))

    total = magnitude.sum(axis=1)
    safe_total = np.where(total > 0.0, total, 1.0)
    centroid = float(np.mean((magnitude * frequencies).sum(axis=1) / safe_total))

    cumulative = np.cumsum(magnitude, axis=1)
    rolloff_point = cumulative[:, -1] * _SPECTRAL_ROLLOFF
    rolloff_index = np.clip(
        (cumulative < rolloff_point[:, None]).sum(axis=1), 0, frequencies.size - 1
    )
    rolloff = float(np.mean(frequencies[rolloff_index]))

    if magnitude.shape[0] >= 2:
        differences = np.diff(magnitude, axis=0)
        flux = float(np.mean(np.sqrt(np.sum(np.square(differences), axis=1))))
    else:
        flux = 0.0

    sign_changes = np.count_nonzero(
        np.diff(voiced_frames >= 0.0, axis=1), axis=1
    )
    zero_crossing_rate = float(np.mean(sign_changes / (frame_length - 1)))

    return {
        "spectral_centroid_hz": round(centroid, 4),
        "spectral_rolloff_hz": round(rolloff, 4),
        "spectral_flux": round(flux, 6),
        "zero_crossing_rate": round(zero_crossing_rate, 6),
    }


def extract_acoustic_features(samples: np.ndarray, sample_rate: int) -> dict:
    """Heuristic acoustic features for rehabilitation speech (numpy + Praat)."""
    signal = to_mono(samples)
    if signal.size == 0 or sample_rate <= 0:
        return _zero_metrics()

    duration = float(signal.size) / float(sample_rate)
    frame_length = max(1, int(round(_FRAME_SECONDS * sample_rate)))
    hop_length = max(1, int(round(_HOP_SECONDS * sample_rate)))

    frames = _frame_signal(signal, frame_length, hop_length)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))

    percentile_90 = float(np.percentile(rms, 90)) if rms.size else 0.0
    threshold = max(MIN_RMS, 0.15 * percentile_90)
    voiced = rms >= threshold

    voiced_ratio = float(voiced.mean())
    pause_ratio = 1.0 - voiced_ratio

    hop_seconds = hop_length / float(sample_rate)
    pause_runs = _run_lengths(~voiced)
    long_pauses = [run * hop_seconds for run in pause_runs if run * hop_seconds >= _MIN_PAUSE_SECONDS]
    mean_pause_seconds = float(np.mean(long_pauses)) if long_pauses else 0.0

    onset_mask = voiced & np.concatenate(([False], voiced[:-1]))
    voiced_onsets = int(np.count_nonzero(onset_mask))
    speaking_rate = voiced_onsets / duration if duration > 0 else 0.0

    f0_values = []
    for index in np.flatnonzero(voiced):
        estimate = _estimate_f0(frames[index], sample_rate)
        if np.isfinite(estimate):
            f0_values.append(estimate)
    if len(f0_values) >= 2:
        f0_array = np.asarray(f0_values, dtype=np.float64)
        mean_f0 = float(f0_array.mean())
        pitch_variability = float(f0_array.std() / mean_f0) if mean_f0 > 0 else 0.0
    else:
        pitch_variability = 0.0

    return {
        "duration_seconds": round(duration, 4),
        "rms_energy": round(float(rms.mean()), 4),
        "voiced_ratio": round(voiced_ratio, 4),
        "pause_ratio": round(pause_ratio, 4),
        "mean_pause_seconds": round(mean_pause_seconds, 4),
        "speaking_rate": round(speaking_rate, 4),
        "pitch_variability": round(pitch_variability, 4),
        **_spectral_metrics(frames, int(sample_rate), frame_length, voiced),
        **_voice_quality_metrics(signal, int(sample_rate)),
    }


def acoustic_findings(metrics: dict) -> list[dict]:
    """Non-diagnostic screening findings derived from acoustic metrics."""
    duration = float(metrics.get("duration_seconds", 0.0) or 0.0)
    rms_energy = float(metrics.get("rms_energy", 0.0) or 0.0)

    if duration < MIN_DURATION_SECONDS or rms_energy < MIN_RMS:
        return [
            {
                "code": "dados_insuficientes",
                "severity": "info",
                "message": (
                    "Audio curto ou silencioso demais para uma triagem acustica "
                    "confiavel; grave novamente em ambiente silencioso."
                ),
                "measured": round(duration, 4),
                "expected": MIN_DURATION_SECONDS,
            }
        ]

    findings: list[dict] = []
    pause_ratio = float(metrics.get("pause_ratio", 0.0) or 0.0)
    mean_pause_seconds = float(metrics.get("mean_pause_seconds", 0.0) or 0.0)
    pitch_variability = float(metrics.get("pitch_variability", 0.0) or 0.0)

    if pause_ratio >= PAUSE_RATIO_LIMIT and rms_energy < LOW_ENERGY_LIMIT:
        findings.append(
            {
                "code": "possivel_fadiga",
                "severity": "warning",
                "message": (
                    "Pausas prolongadas combinadas com baixa energia de voz podem "
                    "indicar fadiga durante o esforco; sugerida revisao do "
                    "fisioterapeuta."
                ),
                "measured": round(rms_energy, 4),
                "expected": LOW_ENERGY_LIMIT,
            }
        )
    if mean_pause_seconds >= MEAN_PAUSE_LIMIT and pause_ratio >= PAUSE_DENSITY_LIMIT:
        findings.append(
            {
                "code": "possivel_desconforto_respiratorio",
                "severity": "warning",
                "message": (
                    "Pausas longas e frequentes podem sugerir desconforto "
                    "respiratorio durante a fala; nao e diagnostico, apenas um sinal "
                    "para avaliacao."
                ),
                "measured": round(mean_pause_seconds, 4),
                "expected": MEAN_PAUSE_LIMIT,
            }
        )
    if pitch_variability >= PITCH_VARIABILITY_LIMIT:
        findings.append(
            {
                "code": "fala_instavel",
                "severity": "warning",
                "message": (
                    "Variabilidade de pitch elevada pode indicar fala instavel; "
                    "resultado experimental de apoio, nao diagnostico."
                ),
                "measured": round(pitch_variability, 4),
                "expected": PITCH_VARIABILITY_LIMIT,
            }
        )
    findings.extend(dysarthria_findings(metrics))
    return findings
