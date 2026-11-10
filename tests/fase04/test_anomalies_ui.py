from pathlib import Path
from urllib.parse import urlsplit

import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from backend.anomalies.service import dashboard
from backend.anomalies.standalone import app


def test_integrated_panel_replay_acknowledge_and_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("ANOMALY_DB_PATH", str(tmp_path / "ui.db"))
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "frontend"))
    with TestClient(app) as client:
        def local_request(method, url, **kwargs):
            kwargs.pop("timeout", None)
            response = client.request(method, urlsplit(url).path, **kwargs)
            adapted = requests.Response()
            adapted.status_code = response.status_code
            adapted._content = response.content
            return adapted

        monkeypatch.setattr(requests, "request", local_request)
        page = AppTest.from_file(str(root / "frontend" / "app.py"), default_timeout=30)
        page.session_state["stage"] = 6
        page.session_state["anomaly_patient"] = "UI-DEMO"
        page.run()
        assert not page.exception
        assert not page.error
        next(b for b in page.button if b.label == "Reproduzir demonstração sintética").click().run()
        assert not page.exception
        assert not page.error
        assert len(dashboard("UI-DEMO")["alerts"]) == 6
        next(b for b in page.button if b.label == "Marcar como visto").click().run()
        assert not page.exception
        assert sum(a["acknowledged_at"] is not None for a in dashboard("UI-DEMO")["alerts"]) == 1
        next(b for b in page.button if b.label == "Reproduzir demonstração sintética").click().run()
        assert not page.exception
        assert len(dashboard("UI-DEMO")["alerts"]) == 6
