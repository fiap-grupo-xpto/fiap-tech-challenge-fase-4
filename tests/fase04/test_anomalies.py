import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest
from fastapi.testclient import TestClient

from backend.anomalies.demo import demo_observations
from backend.anomalies.schemas import Observation, ObservationBatch
from backend.anomalies.service import Conflict, dashboard, ingest
from backend.anomalies.standalone import app


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ANOMALY_DB_PATH", str(tmp_path / "monitor.db"))


def observations(anomalous=True, patient="P1"):
    return ObservationBatch(observations=demo_observations(patient, anomalous=anomalous)).observations


def test_detects_three_domains_with_evidence_and_suppresses_repeated_alerts():
    result = ingest(observations())
    alerts = result["new_alerts"]
    assert {a["kind"] for a in alerts} == {"vitals", "movement", "prescription"}
    assert len(alerts) == 6
    assert all(a["evidence"] and a["requires_human_review"] for a in alerts)
    assert len(dashboard("P1")["observations"]) == 51
    assert not ingest(observations())["new_alerts"]
    assert len(dashboard("P1")["alerts"]) == 6


def test_normal_stream_and_patient_isolation():
    assert not ingest(observations(False))["new_alerts"]
    assert ingest(observations(True, "P2"))["new_alerts"]
    assert not dashboard("P1")["alerts"]


def test_atomic_rollback_on_conflicting_id_or_out_of_order():
    events = observations(False)
    ingest(events[:2])
    conflict = events[0].model_copy(update={"context": "rest"})
    with pytest.raises(Conflict):
        ingest([events[2], conflict])
    assert len(dashboard("P1")["observations"]) == 2
    old = events[0].model_copy(update={"event_id": "different-id"})
    with pytest.raises(Conflict):
        ingest([old])


def test_streaming_and_batch_equivalence():
    first = ingest(observations())
    second = [ingest([event]) for event in observations(patient="P2")]
    assert [a["code"] for a in first["new_alerts"]] == [a["code"] for r in second for a in r["new_alerts"]]


def test_single_spike_does_not_alert_and_future_does_not_affect_past():
    events = [e for e in observations(False) if e.kind == "vitals"]
    events[8] = events[8].model_copy(update={"metrics": {**events[8].metrics, "heart_rate": 150}})
    result = ingest(events)
    assert not result["new_alerts"]
    assert result["results"][8]["state"] == "candidate"
    assert result["results"][7]["state"] == "normal"


@pytest.mark.parametrize("change", ["gap", "context", "irregular"])
def test_movement_resets_baseline_on_noncomparable_history(change):
    events = [e for e in observations(False) if e.kind == "movement"][:9]
    ingest(events[:8])
    event = events[8].model_copy(update={"metrics": {"activity_fraction": 0.0}})
    if change == "context":
        event.context = "rest"
    else:
        event.timestamp += timedelta(minutes=30 if change == "gap" else 1)
    result = ingest([event])
    assert result["results"][0]["state"] == "warming_up"
    assert not result["new_alerts"]


def test_prescription_unit_change_is_not_numeric_dose_comparison():
    events = [e for e in observations(False) if e.kind == "prescription"]
    events[1].prescription.unit = "g"
    events[1].prescription.dose = 0.01
    result = ingest(events[:2])
    changes = result["new_alerts"][0]["evidence"]["changes"]
    assert [c["field"] for c in changes] == ["unit"]


def test_concurrent_retry_is_idempotent():
    event = observations()[0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: ingest([event]), range(2)))
    assert sum(r["results"][0]["duplicate"] for r in results) == 1
    assert len(dashboard("P1")["observations"]) == 1


def test_api_validation_and_acknowledgment():
    with TestClient(app) as client:
        demo = client.get("/anomalies/demo").json()
        response = client.post("/anomalies/observations", json={"observations": demo["observations"]})
        assert response.status_code == 200
        alert = response.json()["new_alerts"][0]
        url = f"/anomalies/alerts/{alert['id']}/acknowledge"
        first = client.post(url).json()
        assert first["acknowledged_at"]
        assert client.post(url).json()["acknowledged_at"] == first["acknowledged_at"]
        assert client.post("/anomalies/alerts/missing/acknowledge").status_code == 404
        payload = demo["observations"][0]
        payload["timestamp"] = "2026-01-01T00:00:00"
        assert client.post("/anomalies/observations", json={"observations": [payload]}).status_code == 422
        assert client.get("/anomalies/patients/DEMO-001").json()["alerts"]


def test_movement_tolerates_collector_jitter():
    events = [e for e in observations(False) if e.kind == "movement"][:9]
    events = [e.model_copy(update={"timestamp": e.timestamp + timedelta(seconds=10 * (i % 2))}) for i, e in enumerate(events)]
    assert ingest(events)["results"][-1]["state"] == "normal"


class _Webhook(BaseHTTPRequestHandler):
    received = []

    def do_POST(self):
        _Webhook.received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def wait_for_notifications(client, patient):
    """Delivery is asynchronous; HTTP response does not imply worker completion."""
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        alerts = client.get(f"/anomalies/patients/{patient}").json()["alerts"]
        if alerts and all(a["notifications"] for a in alerts):
            return alerts
        time.sleep(.02)
    pytest.fail("Notifications were not recorded before the deadline")


def test_alerts_are_delivered_to_team_webhook(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Webhook)
    Thread(target=server.serve_forever, daemon=True).start()
    _Webhook.received.clear()
    monkeypatch.setenv("ALERT_WEBHOOK_URL", f"http://127.0.0.1:{server.server_port}/hook")
    try:
        with TestClient(app) as client:
            demo = client.get("/anomalies/demo").json()
            client.post("/anomalies/observations", json={"observations": demo["observations"]})
            alerts = wait_for_notifications(client, "DEMO-001")
    finally:
        server.shutdown()
        server.server_close()
    assert len(_Webhook.received) == 6
    assert all("revisão necessária" in body["text"] and "DEMO-001" in body["text"] for body in _Webhook.received)
    assert all(a["notifications"][0]["status"] == "sent" for a in alerts)


def test_delivery_failure_or_missing_webhook_keeps_alert(monkeypatch):
    for url, status in (("", "panel_only"), ("http://127.0.0.1:9/unreachable", "failed")):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", url)
        patient = "NOTIFY-" + status
        with TestClient(app) as client:
            demo = client.get("/anomalies/demo", params={"patient_id": patient}).json()
            assert client.post("/anomalies/observations", json={"observations": demo["observations"]}).status_code == 200
            alerts = wait_for_notifications(client, patient)
        assert len(alerts) == 6
        assert {a["notifications"][0]["status"] for a in alerts} == {status}


@pytest.mark.parametrize("metrics", [{},{"activity_fraction": -0.1}, {"activity_fraction": float("nan")},
                                     {"activity_fraction": 0.5, "extra": 1}])
def test_rejects_missing_or_invalid_movement_measurement(metrics):
    with pytest.raises(ValueError):
        Observation(event_id="e", patient_id="p", timestamp=datetime.now(timezone.utc), kind="movement", metrics=metrics)
