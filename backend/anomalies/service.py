import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from backend.anomalies.detector import VERSION, candidates
from backend.anomalies.schemas import Observation


class Conflict(ValueError):
    pass


@contextmanager
def connection():
    path = Path(os.getenv("ANOMALY_DB_PATH", str(Path(__file__).resolve().parents[1] / "data" / "anomalies.db")))
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS observations (
                event_id TEXT PRIMARY KEY, patient_id TEXT NOT NULL,
                stream TEXT NOT NULL, timestamp TEXT NOT NULL,
                payload TEXT NOT NULL, result TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS observation_stream
                ON observations(patient_id, stream, timestamp);
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, timestamp TEXT NOT NULL,
                payload TEXT NOT NULL, acknowledged_at TEXT);
            CREATE INDEX IF NOT EXISTS alert_patient ON alerts(patient_id, timestamp);
            CREATE TABLE IF NOT EXISTS notifications (
                alert_id TEXT NOT NULL, status TEXT NOT NULL,
                detail TEXT NOT NULL, at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS notification_alert ON notifications(alert_id);
            CREATE TABLE IF NOT EXISTS notification_outbox (
                alert_id TEXT PRIMARY KEY REFERENCES alerts(id),
                status TEXT NOT NULL DEFAULT 'pending',
                lease_token TEXT, lease_until REAL NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0);
        """)
        yield db
    finally:
        db.close()


def ingest(observations: list[Observation]) -> dict:
    """Atomic batch, causal order per patient/stream; retries have no side effects."""
    results, new_alerts = [], []
    with connection() as db:
        with db:
            db.execute("BEGIN IMMEDIATE")
            for event in observations:
                payload = event.model_dump_json()
                existing = db.execute("SELECT payload, result FROM observations WHERE event_id=?", (event.event_id,)).fetchone()
                if existing:
                    if json.loads(existing["payload"]) != json.loads(payload):
                        raise Conflict("event_id já utilizado com outro conteúdo")
                    results.append({**json.loads(existing["result"]), "duplicate": True})
                    continue
                rows = db.execute(
                    "SELECT payload, result FROM observations WHERE patient_id=? AND stream=? ORDER BY timestamp DESC LIMIT 13",
                    (event.patient_id, event.stream),
                ).fetchall()[::-1]
                history = [Observation.model_validate_json(row["payload"]) for row in rows]
                if history and event.timestamp <= history[-1].timestamp:
                    raise Conflict("Envie eventos em ordem crescente por paciente e tipo/medicamento; horários iguais não são permitidos")
                found, state = candidates(event, history)
                previous_candidates = set(json.loads(rows[-1]["result"])["candidate_codes"]) if rows else set()
                active = set(json.loads(rows[-1]["result"])["active_codes"]) if rows else set()
                # A gap/context change produces no candidates and resets persistence.
                codes = {f["code"] for f in found}
                confirmed = codes if event.kind == "prescription" else codes & previous_candidates
                # Prescriptions are discrete changes: every new change needs review.
                emit = confirmed if event.kind == "prescription" else confirmed - active
                event_alerts = []
                for finding in found:
                    if finding["code"] not in emit:
                        continue
                    alert = {"id": str(uuid.uuid4()), "patient_id": event.patient_id,
                             "event_id": event.event_id, "timestamp": event.timestamp.isoformat(),
                             "kind": event.kind, **finding, "method": VERSION,
                             "requires_human_review": True,
                             "acknowledged_at": None}
                    db.execute("INSERT INTO alerts(id, patient_id, timestamp, payload) VALUES(?,?,?,?)",
                               (alert["id"], event.patient_id, alert["timestamp"], json.dumps(alert)))
                    # Commit the delivery intent atomically with the alert.
                    db.execute("INSERT INTO notification_outbox(alert_id) VALUES(?)", (alert["id"],))
                    event_alerts.append(alert)
                result = {"event_id": event.event_id, "patient_id": event.patient_id,
                          "state": "anomaly" if confirmed else state,
                          "candidate_codes": sorted(codes), "active_codes": sorted(confirmed),
                          "findings": found, "alerts": event_alerts, "duplicate": False}
                db.execute("INSERT INTO observations VALUES(?,?,?,?,?,?)",
                           (event.event_id, event.patient_id, event.stream, event.timestamp.isoformat(), payload, json.dumps(result)))
                results.append(result)
                new_alerts.extend(event_alerts)
    return {"results": results, "new_alerts": new_alerts, "method": VERSION,
            "warning": "Protótipo experimental. Desvios estatísticos não são diagnósticos; requer revisão humana."}


def dashboard(patient_id: str) -> dict:
    with connection() as db:
        events = db.execute("SELECT payload, result FROM observations WHERE patient_id=? ORDER BY timestamp DESC LIMIT 500", (patient_id,)).fetchall()
        rows = db.execute("SELECT payload, acknowledged_at FROM alerts WHERE patient_id=? ORDER BY timestamp DESC LIMIT 500", (patient_id,)).fetchall()
        notices = {}
        deliveries = {row["alert_id"]: {"status": row["status"], "attempts": row["attempts"]}
                      for row in db.execute(
                          "SELECT o.alert_id, o.status, o.attempts FROM notification_outbox o "
                          "JOIN alerts a ON a.id=o.alert_id WHERE a.patient_id=?", (patient_id,))}
        for row in db.execute("SELECT n.alert_id, n.status, n.detail, n.at FROM notifications n JOIN alerts a ON a.id = n.alert_id "
                              "WHERE a.patient_id=? ORDER BY n.at", (patient_id,)):
            notices.setdefault(row["alert_id"], []).append({"status": row["status"], "detail": row["detail"], "at": row["at"]})
    alerts = [{**json.loads(row["payload"]), "acknowledged_at": row["acknowledged_at"]} for row in rows]
    return {"patient_id": patient_id,
            "observations": [{**json.loads(row["payload"]), "analysis": json.loads(row["result"])} for row in reversed(events)],
            "alerts": [{**alert, "notifications": notices.get(alert["id"], []),
                        "delivery": deliveries.get(alert["id"])} for alert in alerts],
            "limit": 500}


def acknowledge(alert_id: str) -> dict | None:
    with connection() as db:
        with db:
            now = datetime.now(timezone.utc).isoformat()
            db.execute("UPDATE alerts SET acknowledged_at=COALESCE(acknowledged_at, ?) WHERE id=?", (now, alert_id))
            row = db.execute("SELECT payload, acknowledged_at FROM alerts WHERE id=?", (alert_id,)).fetchone()
    return {**json.loads(row["payload"]), "acknowledged_at": row["acknowledged_at"]} if row else None
