"""Combine rehabilitation findings and confirmed temporal alerts for one case."""

from datetime import datetime
from typing import Literal

from backend.anomalies.service import alerts_in_window
from backend.rehab_audio.report import build_support_report
from backend.rehab_audio.terms import extract_critical_terms


def build_case_report(
    patient_id: str,
    start: datetime,
    end: datetime,
    clinical_note: str,
    audio: dict,
    video: dict,
    mode: Literal["upload", "demo"],
) -> dict:
    alerts = alerts_in_window(patient_id, start, end)
    note_terms = extract_critical_terms(clinical_note)
    text_signals = [term["term"] for term in note_terms if term["severity"] == "warning"]
    voice_signals = [
        finding["code"] for finding in audio["acoustic_findings"]
        if finding["severity"] == "warning"
    ]
    if (audio.get("voice_alteration") or {}).get("flagged"):
        voice_signals.append("alteracao_vocal_classificador")
    pose_signals = [
        finding["code"] for finding in video["deviation_report"]["findings"]
        if finding["severity"] == "warning"
    ]
    if video["classification"] == "incorrect":
        pose_signals.append("repeticao_possivelmente_incorreta")

    signals = {"text": text_signals, "audio": voice_signals, "video": pose_signals}
    sources = [source for source, evidence in signals.items() if evidence]
    if alerts and len(sources) >= 2:
        level = "prioritized_review"
        reason = "Há alertas confirmados na janela e sinais em pelo menos duas modalidades."
    elif alerts or sources:
        level = "review"
        reason = "Há achados para revisar na sessão ou no monitoramento."
    else:
        level = "no_signal"
        reason = "As regras do protótipo não sinalizaram achados nesta janela."

    return {
        "patient_id": patient_id,
        "window": {"start": start, "end": end},
        "mode": mode,
        "text": {"clinical_note": clinical_note, "critical_terms": note_terms},
        "audio": audio,
        "video": video,
        "confirmed_alerts": alerts,
        "signals": signals,
        "review": {"level": level, "reason": reason, "sources_with_signals": sources},
        "requires_human_review": True,
    }


def demo_media_reports() -> tuple[str, dict, dict]:
    note = "Registro simulado de fisioterapia: paciente relatou dor durante o agachamento."
    transcript = "Estou com dor no joelho e cansaço quando agacho."
    audio = build_support_report(
        {"transcript": transcript},
        {
            "sentiment": {"label": "negative", "positive": 0.02, "neutral": 0.08, "negative": 0.90},
            "key_phrases": ["dor no joelho", "cansaço"],
            "entities": [],
        },
        {"duration_seconds": 8.0, "rms_energy": 0.01},
        [{
            "code": "possivel_fadiga", "severity": "warning",
            "message": "Indicador vocal simulado de possível fadiga.",
            "measured": 0.01, "expected": 0.02,
        }],
        extract_critical_terms(transcript),
    )
    video = {
        "exercise_id": 6,
        "classification": "incorrect",
        "deviation_report": {
            "deviation_count": 1,
            "summary": "Amplitude reduzida no agachamento (resultado simulado).",
            "findings": [{
                "code": "amplitude_reduzida", "severity": "warning",
                "message": "Amplitude do joelho abaixo da referência experimental.",
                "measured": 25, "expected": 40,
            }],
        },
    }
    return note, audio, video
