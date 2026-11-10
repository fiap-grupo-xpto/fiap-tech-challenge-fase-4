import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from backend.anomalies import notifier
from backend.anomalies.api import submit
from backend.anomalies.demo import demo_observations
from backend.anomalies.schemas import ObservationBatch
from backend.anomalies.service import Conflict, connection, dashboard, ingest
from backend.anomalies.standalone import app


def batch():
    return ObservationBatch(observations=demo_observations("RECOVERY"))


def test_restart_recovers_committed_alerts_without_http_task(monkeypatch):
    pending = BackgroundTasks()
    response = submit(batch(), pending)  # Simulate crash before task execution.
    assert len(response["new_alerts"]) == 6
    assert {a["delivery"]["status"] for a in dashboard("RECOVERY")["alerts"]} == {"pending"}
    sent = []
    monkeypatch.setattr(notifier, "_send", lambda a: (sent.append(a["id"]) or ("sent", "fake transport")))
    with TestClient(app) as client:  # New application lifespan recovers the queue.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            alerts = client.get("/anomalies/patients/RECOVERY").json()["alerts"]
            if all(a["delivery"]["status"] == "sent" for a in alerts):
                break
            time.sleep(.02)
        assert len(sent) == 6
        assert all(a["delivery"]["status"] == "sent" for a in alerts)
        assert not client.post("/anomalies/observations", json=batch().model_dump(mode="json")).json()["new_alerts"]
    assert len(set(sent)) == len(sent) == 6


def test_interrupted_inflight_delivery_is_recovered_after_lease(monkeypatch):
    alert = ingest(batch().observations)["new_alerts"][0]
    def crash(_):
        raise SystemExit("simulated process interruption")
    monkeypatch.setattr(notifier, "_send", crash)
    with pytest.raises(SystemExit):
        notifier.process_one(alert["id"])
    assert not notifier.process_one(alert["id"])  # Active claim cannot be stolen.
    with connection() as db:
        with db:
            db.execute("UPDATE notification_outbox SET lease_until=0 WHERE alert_id=?", (alert["id"],))
    monkeypatch.setattr(notifier, "_send", lambda _: ("sent", "fake transport"))
    assert notifier.process_one(alert["id"])
    recovered = next(a for a in dashboard("RECOVERY")["alerts"] if a["id"] == alert["id"])
    assert recovered["delivery"] == {"status": "sent", "attempts": 2}
    assert len(recovered["notifications"]) == 1


def test_concurrent_workers_claim_each_alert_only_once(monkeypatch):
    alerts = ingest(batch().observations)["new_alerts"]
    sent = []
    def send(alert):
        time.sleep(.01)
        sent.append(alert["id"])
        return "sent", "fake transport"
    monkeypatch.setattr(notifier, "_send", send)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: notifier.deliver(alerts), range(4)))
    assert len(sent) == len(set(sent)) == 6
    assert all(len(a["notifications"]) == 1 for a in dashboard("RECOVERY")["alerts"])


@pytest.mark.parametrize("url", ["invalid-webhook-url", "https://[invalid", "file:///local/path"])
def test_malformed_url_records_failure_for_every_alert(monkeypatch, url):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", url)
    def forbidden_network(*args, **kwargs):
        pytest.fail("Invalid URLs must not reach the transport")
    monkeypatch.setattr(notifier.urllib.request, "urlopen", forbidden_network)
    alerts = ingest(batch().observations)["new_alerts"]
    notifier.deliver(alerts)
    data = dashboard("RECOVERY")["alerts"]
    assert len(data) == 6
    assert all(a["delivery"]["status"] == "failed" for a in data)
    assert all(a["notifications"][0]["status"] == "failed" for a in data)
    notifier.deliver(alerts)
    assert all(a["delivery"]["attempts"] == 1 for a in dashboard("RECOVERY")["alerts"])


def test_outbox_rolls_back_with_conflicting_batch():
    events = batch().observations
    conflict = events[0].model_copy(update={"context": "rest"})
    with pytest.raises(Conflict):
        ingest(events + [conflict])
    with connection() as db:
        assert db.execute("SELECT COUNT(*) FROM notification_outbox").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 0


def test_completed_delivery_survives_new_worker(monkeypatch):
    alerts = ingest(batch().observations)["new_alerts"]
    notifier.deliver(alerts)  # No webhook: explicitly terminal panel_only.
    monkeypatch.setattr(notifier, "_send", lambda _: pytest.fail("Completed deliveries must not be repeated"))
    with TestClient(app):
        notifier.deliver(alerts)
    assert all(a["delivery"]["status"] == "panel_only" for a in dashboard("RECOVERY")["alerts"])
