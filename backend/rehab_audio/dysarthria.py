"""Heuristic screening for motor-speech (dysarthria) indicators.

Non-diagnostic. Uses standard voice-quality cut-offs (Praat/MDVP norms) computed
over the voiced portion of a recording: local jitter, local shimmer and the
harmonics-to-noise ratio. Rhythm markers (rate and pauses) reinforce the flag in
full session audio, but the voice-quality core is what the open-data evaluation
exercises.
"""

from __future__ import annotations

DYSARTHRIA_FEATURE_KEYS = ("jitter_local", "shimmer_local", "hnr")

# Bounds for normophonia (Praat local jitter/shimmer percentages, MDVP HNR in dB).
JITTER_LIMIT_PERCENT = 1.04
SHIMMER_LIMIT_PERCENT = 3.81
HNR_LIMIT_DB = 20.0
MIN_DEVIANT_MARKERS = 2
# Default cut-offs keyed by marker; can be overridden (e.g. data-calibrated).
DEFAULT_LIMITS = {
    "jitter_local": JITTER_LIMIT_PERCENT,
    "shimmer_local": SHIMMER_LIMIT_PERCENT,
    "hnr": HNR_LIMIT_DB,
}


def _positive(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 else 0.0


def _deviations(metrics: dict, limits: dict) -> list[tuple[str, float, float]]:
    """Return ``(marker, measured, limit)`` for each deviant voice-quality marker."""
    jitter = _positive(metrics.get("jitter_local"))
    shimmer = _positive(metrics.get("shimmer_local"))
    hnr = _positive(metrics.get("hnr"))

    jitter_limit = float(limits["jitter_local"])
    shimmer_limit = float(limits["shimmer_local"])
    hnr_limit = float(limits["hnr"])

    deviations: list[tuple[str, float, float]] = []
    if jitter and jitter >= jitter_limit:
        deviations.append(("jitter_local", jitter, jitter_limit))
    if shimmer and shimmer >= shimmer_limit:
        deviations.append(("shimmer_local", shimmer, shimmer_limit))
    if hnr and hnr <= hnr_limit:
        deviations.append(("hnr", hnr, hnr_limit))
    return deviations


def dysarthria_findings(metrics: dict, limits: dict | None = None) -> list[dict]:
    """Non-diagnostic dysarthria screening finding from voice-quality markers.

    ``limits`` optionally overrides the default normophonia cut-offs (see
    ``DEFAULT_LIMITS``) so the harness can test data-calibrated thresholds.
    """
    deviations = _deviations(metrics, limits or DEFAULT_LIMITS)
    if len(deviations) < MIN_DEVIANT_MARKERS:
        return []

    marker, measured, limit = max(
        deviations,
        key=lambda item: abs(item[1] - item[2]) / item[2] if item[2] else 0.0,
    )
    return [
        {
            "code": "possivel_disartria",
            "severity": "warning",
            "message": (
                "Instabilidade vocal (jitter/shimmer elevados ou relacao "
                "harmonico-ruido baixa) pode indicar disartria; resultado "
                "experimental de apoio, nao diagnostico."
            ),
            "measured": round(measured, 4),
            "expected": limit,
        }
    ]
