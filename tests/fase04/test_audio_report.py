from backend.rehab_audio.report import build_support_report


CONTRACT_KEYS = {
    "status",
    "transcript",
    "segments",
    "critical_terms",
    "sentiment",
    "azure_key_phrases",
    "azure_entities",
    "acoustic_findings",
    "metrics",
    "voice_alteration",
    "risk_flags",
    "requires_human_review",
    "warning",
    "method",
    "report_version",
}


def _sample_inputs():
    transcription = {
        "transcript": "repita o movimento, estou com dor",
        "segments": [
            {
                "start": 0.0,
                "end": 1.5,
                "text": "repita o movimento",
                "confidence": 0.9,
            }
        ],
        "language": "pt-BR",
    }
    text_analysis = {
        "sentiment": {
            "label": "negative",
            "positive": 0.1,
            "neutral": 0.2,
            "negative": 0.7,
        },
        "key_phrases": ["dor"],
        "entities": [{"text": "joelho", "category": "BodyPart"}],
    }
    metrics = {
        "duration_seconds": 30.0,
        "rms_energy": 0.01,
        "voiced_ratio": 0.3,
        "pause_ratio": 0.7,
        "mean_pause_seconds": 1.5,
        "speaking_rate": 0.5,
        "pitch_variability": 0.1,
    }
    findings = [
        {
            "code": "possivel_fadiga",
            "severity": "warning",
            "message": "mensagem",
            "measured": 0.01,
            "expected": 0.05,
        }
    ]
    terms = [
        {"term": "dor", "category": "sintoma", "count": 1, "severity": "warning"},
        {"term": "repita", "category": "instrucao", "count": 1, "severity": "info"},
    ]
    return transcription, text_analysis, metrics, findings, terms


def test_build_support_report_returns_full_contract():
    report = build_support_report(*_sample_inputs())

    assert set(report) == CONTRACT_KEYS
    assert report["status"] == "success"
    assert report["transcript"] == "repita o movimento, estou com dor"
    assert report["requires_human_review"] is True
    assert report["report_version"] == "1.0"


def test_build_support_report_includes_azure_insights():
    report = build_support_report(*_sample_inputs())

    assert report["azure_key_phrases"] == ["dor"]
    assert report["azure_entities"] == [
        {"text": "joelho", "category": "BodyPart"}
    ]


def test_build_support_report_risk_flags_union():
    report = build_support_report(*_sample_inputs())

    assert report["risk_flags"] == ["possivel_fadiga", "sintoma"]


def test_build_support_report_framing_and_metrics():
    report = build_support_report(*_sample_inputs())

    assert "nao e diagnostico" in report["warning"]
    assert "nao validadas clinicamente" in report["method"]
    assert set(report["metrics"]) == {
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
    assert report["sentiment"]["label"] == "negative"


def test_build_support_report_voice_alteration_defaults_to_none():
    report = build_support_report(*_sample_inputs())

    assert report["voice_alteration"] is None


def test_build_support_report_surfaces_voice_alteration():
    voice_alteration = {
        "probability_pathological_voice": 0.82,
        "threshold": 0.5,
        "flagged": True,
        "model": "StandardScaler + LogisticRegression",
        "dataset": "VOICED",
        "validation_roc_auc": 0.6,
        "feature_version": "rehab-audio-acoustic-v1",
    }

    report = build_support_report(*_sample_inputs(), voice_alteration=voice_alteration)

    assert report["voice_alteration"] == voice_alteration
