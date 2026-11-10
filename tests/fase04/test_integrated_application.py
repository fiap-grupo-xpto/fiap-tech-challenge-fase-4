"""Shared application integration; external/model limitations are explicit."""
import io
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pytest
import requests
import soundfile as sf
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import backend.main as main
from backend.anomalies.demo import demo_observations
from backend.assistant import llm_adapter
from backend.rehab_audio import service as audio_service
from backend.rehab_video import service as video_service


@pytest.fixture
def integrated_client(monkeypatch, tmp_path):
    monkeypatch.setenv("ITEM1_PRELOAD", "false")
    monkeypatch.setenv("ANOMALY_DB_PATH", str(tmp_path / "anomalies.db"))
    monkeypatch.setenv("ASSISTANT_DB_PATH", str(tmp_path / "hospital.db"))
    monkeypatch.setenv("ASSISTANT_AUDIT_FALLBACK", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("ITEM1_MODEL_DIR", str(tmp_path / "missing-model"))
    monkeypatch.setenv("ITEM1_TOKENIZER_DIR", str(tmp_path / "missing-tokenizer"))
    for key in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_LANGUAGE_KEY",
                "AZURE_LANGUAGE_ENDPOINT", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(llm_adapter, "_cached_item1_provider", None)
    monkeypatch.setattr(llm_adapter, "_cached_gemini_provider", None)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(main, "UPLOAD_DIR", uploads)
    speech_factory = audio_service._get_stt_client
    language_factory = audio_service._get_language_client
    speech_factory.cache_clear()
    language_factory.cache_clear()
    with TestClient(main.app) as client:
        yield client
    speech_factory.cache_clear()
    language_factory.cache_clear()


def wav_bytes():
    output = io.BytesIO()
    t = np.arange(32000) / 16000
    sf.write(output, 0.2 * np.sin(2 * np.pi * 220 * t), 16000, format="WAV", subtype="PCM_16")
    return output.getvalue()


def test_full_app_keeps_anomalies_available_when_other_modules_are_unavailable(integrated_client, monkeypatch, tmp_path):
    client = integrated_client
    # Actual video service with deliberately missing artifact: no fake response.
    def missing_video(path, exercise_id, start_frame, end_frame):
        return video_service.analyze_repetition(path, exercise_id, start_frame, end_frame,
                                                model_path=tmp_path / "missing.joblib")

    monkeypatch.setattr(main, "analyze_repetition", missing_video)
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/anomalies/observations", "/assistant/query", "/analyze-rehabilitation-audio",
            "/analyze-rehabilitation-video", "/annotate-rehabilitation-video"} <= set(paths)
    batch = {"observations": demo_observations("INTEGRATED")}
    result = client.post("/anomalies/observations", json=batch)
    assert result.status_code == 200
    assert len(result.json()["new_alerts"]) == 6
    audio = client.post("/analyze-rehabilitation-audio",
                        files={"audio": ("synthetic.wav", wav_bytes(), "audio/wav")})
    assert audio.status_code == 503
    assert "Credenciais" in audio.json()["detail"]
    video = client.post("/analyze-rehabilitation-video",
                        files={"video": ("sample.mp4", b"placeholder", "video/mp4")}, data={"exercise_id": "1"})
    assert video.status_code == 503
    assert "not trained" in video.json()["detail"]
    assistant = client.post("/assistant/query", json={"patient_id": "P001", "question": "What should be reviewed next?"})
    assert assistant.status_code == 200
    assert assistant.json()["requires_human_review"] is True
    assert assistant.json()["status"] == "success"
    assert "determin" in assistant.json()["message"]
    assert list(main.UPLOAD_DIR.iterdir()) == []
    assert not client.post("/anomalies/observations", json=batch).json()["new_alerts"]
    alert = result.json()["new_alerts"][0]
    assert client.post(f"/anomalies/alerts/{alert['id']}/acknowledge").status_code == 200
    assert len(client.get("/anomalies/patients/INTEGRATED").json()["alerts"]) == 6


def test_actual_audio_pipeline_with_only_azure_clients_simulated(integrated_client, monkeypatch):
    class Speech:
        language = "pt-BR"

        def transcribe(self, path):
            assert Path(path).is_file()
            return {"transcript": "Estou com dor e cansaço.", "segments": [], "language": "pt-BR"}

    class Language:
        def analyze(self, text):
            assert "dor" in text
            return {"sentiment": {"label": "negative", "positive": 0, "neutral": 0.1, "negative": 0.9}}

    monkeypatch.setattr(audio_service, "_get_stt_client", lambda: Speech())
    monkeypatch.setattr(audio_service, "_get_language_client", lambda: Language())
    response = integrated_client.post("/analyze-rehabilitation-audio",
                                      files={"audio": ("synthetic.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 200
    report = response.json()
    assert report["transcript"] == "Estou com dor e cansaço."
    assert report["metrics"]["duration_seconds"] == pytest.approx(2)
    assert report["critical_terms"]
    assert report["requires_human_review"]
    assert list(main.UPLOAD_DIR.iterdir()) == []


def test_navigation_and_anomaly_replay_against_shared_app(integrated_client, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "frontend"))

    def request(method, url, **kwargs):
        kwargs.pop("timeout", None)
        response = integrated_client.request(method, urlsplit(url).path, **kwargs)
        adapted = requests.Response()
        adapted.status_code = response.status_code
        adapted._content = response.content
        return adapted

    monkeypatch.setattr(requests, "request", request)
    page = AppTest.from_file(str(root / "frontend" / "app.py"), default_timeout=30)
    page.session_state["stage"] = 6
    page.session_state["anomaly_patient"] = "UI-INTEGRATED"
    page.run()
    assert not page.exception and not page.error
    next(b for b in page.button if b.label == "Reproduzir demonstração sintética").click().run()
    assert not page.exception and not page.error
    assert len(integrated_client.get("/anomalies/patients/UI-INTEGRATED").json()["alerts"]) == 6
    for option in page.sidebar.radio[0].options:
        page.sidebar.radio[0].set_value(option).run()
        assert not page.exception, option
        assert not page.error, option
