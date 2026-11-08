from fastapi.testclient import TestClient

import backend.main as backend_main


def _stub_report(*_args, **_kwargs) -> dict:
    return {
        "status": "success",
        "transcript": "repita o movimento",
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "repita",
                "confidence": 0.9,
            }
        ],
        "critical_terms": [],
        "sentiment": {
            "label": "neutral",
            "positive": 0.0,
            "neutral": 1.0,
            "negative": 0.0,
        },
        "acoustic_findings": [],
        "metrics": {
            "duration_seconds": 3.0,
            "rms_energy": 0.1,
            "voiced_ratio": 0.9,
            "pause_ratio": 0.1,
            "mean_pause_seconds": 0.0,
            "speaking_rate": 2.0,
            "pitch_variability": 0.1,
        },
        "risk_flags": [],
        "requires_human_review": True,
        "warning": "aviso",
        "method": "metodo",
        "report_version": "1.0",
    }


def test_rehabilitation_audio_endpoint_returns_analysis(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(backend_main, "analyze_session_audio", _stub_report)

    with TestClient(backend_main.app) as client:
        response = client.post(
            "/analyze-rehabilitation-audio",
            files={"audio": ("sample.wav", b"audio-data", "audio/wav")},
            data={"language": "pt-BR"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["requires_human_review"] is True
    assert list(tmp_path.iterdir()) == []


def test_rehabilitation_audio_endpoint_rejects_unsupported_extension(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "UPLOAD_DIR", tmp_path)

    with TestClient(backend_main.app) as client:
        response = client.post(
            "/analyze-rehabilitation-audio",
            files={"audio": ("sample.ogg", b"audio-data", "audio/ogg")},
        )

    assert response.status_code == 415
    assert ".wav" in response.json()["detail"].lower()


def test_rehabilitation_audio_endpoint_rejects_oversized_upload(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(backend_main, "MAX_REHAB_AUDIO_BYTES", 4)
    monkeypatch.setattr(backend_main, "analyze_session_audio", _stub_report)

    with TestClient(backend_main.app) as client:
        response = client.post(
            "/analyze-rehabilitation-audio",
            files={"audio": ("sample.wav", b"12345", "audio/wav")},
        )

    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []
