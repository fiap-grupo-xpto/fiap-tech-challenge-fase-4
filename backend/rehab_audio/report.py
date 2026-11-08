from __future__ import annotations

from backend.rehab_audio.acoustics import METRIC_KEYS


REPORT_VERSION = "1.0"

_WARNING = (
    "Resultado experimental de apoio a reabilitacao; nao e diagnostico e exige "
    "revisao do fisioterapeuta."
)
_METHOD = (
    "Azure Speech to Text + Azure AI Language + regras acusticas heuristicas de "
    "triagem (nao validadas clinicamente)."
)


def _num(value) -> float:
    """Coerce a possibly-missing numeric field to a float (0.0 when absent)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_support_report(
    transcription: dict,
    text_analysis: dict,
    metrics: dict,
    findings: list[dict],
    terms: list[dict],
    voice_alteration: dict | None = None,
) -> dict:
    """Assemble the rehabilitation audio support report (non-diagnostic)."""
    segments = transcription.get("segments") or []
    sentiment = text_analysis.get("sentiment") or {}

    risk_flags = sorted(
        {item["code"] for item in findings if item.get("severity") == "warning"}
        | {
            item["category"]
            for item in terms
            if item.get("severity") == "warning"
        }
    )

    return {
        "status": "success",
        "transcript": transcription.get("transcript", ""),
        "segments": [
            {
                "start": _num(segment.get("start")),
                "end": _num(segment.get("end")),
                "text": segment.get("text", ""),
                "confidence": _num(segment.get("confidence")),
            }
            for segment in segments
        ],
        "critical_terms": [
            {
                "term": term.get("term", ""),
                "category": term.get("category", ""),
                "count": int(term.get("count", 0) or 0),
                "severity": term.get("severity", "info"),
            }
            for term in terms
        ],
        "sentiment": {
            "label": sentiment.get("label", "neutral"),
            "positive": _num(sentiment.get("positive")),
            "neutral": _num(sentiment.get("neutral")),
            "negative": _num(sentiment.get("negative")),
        },
        "azure_key_phrases": [
            str(phrase) for phrase in (text_analysis.get("key_phrases") or [])
        ],
        "azure_entities": [
            {
                "text": str(entity.get("text", "")),
                "category": str(entity.get("category", "")),
            }
            for entity in (text_analysis.get("entities") or [])
        ],
        "acoustic_findings": [
            {
                "code": finding.get("code", ""),
                "severity": finding.get("severity", "info"),
                "message": finding.get("message", ""),
                "measured": _num(finding.get("measured")),
                "expected": _num(finding.get("expected")),
            }
            for finding in findings
        ],
        "metrics": {key: _num(metrics.get(key)) for key in METRIC_KEYS},
        "voice_alteration": (
            None
            if voice_alteration is None
            else {
                "probability_pathological_voice": _num(
                    voice_alteration.get("probability_pathological_voice")
                ),
                "threshold": _num(voice_alteration.get("threshold")),
                "flagged": bool(voice_alteration.get("flagged", False)),
                "model": str(voice_alteration.get("model", "")),
                "dataset": str(voice_alteration.get("dataset", "")),
                "validation_roc_auc": voice_alteration.get("validation_roc_auc"),
                "feature_version": str(voice_alteration.get("feature_version", "")),
            }
        ),
        "risk_flags": risk_flags,
        "requires_human_review": True,
        "warning": _WARNING,
        "method": _METHOD,
        "report_version": REPORT_VERSION,
    }
