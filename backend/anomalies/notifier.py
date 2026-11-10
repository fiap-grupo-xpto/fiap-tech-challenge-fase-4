"""Delivery of confirmed alerts to the care team. Runs after the ingest commit."""
import json
import os
import urllib.request
from urllib.parse import urlsplit
import logging
import threading
import time
import uuid
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timezone

from backend.anomalies.service import connection

TIMEOUT_SECONDS = 5
LEASE_SECONDS = 60
POLL_SECONDS = 5
logger = logging.getLogger(__name__)


def message(alert: dict) -> str:
    evidence = alert["evidence"]
    if alert["kind"] == "prescription":
        detail = ", ".join(f"{c['field']}: {c['previous']} -> {c['current']}" for c in evidence["changes"])
        detail = f"{evidence['medication']} ({detail})"
    else:
        detail = (f"{evidence['metric']} = {evidence['value']:.1f} "
                  f"(mediana do histórico {evidence['baseline_median']:.1f}, escore {evidence['robust_score']:.1f})")
    return (f"[ALERTA - revisão necessária] Paciente {alert['patient_id']} | {alert['timestamp']} UTC\n"
            f"{alert['message']}\n{detail}\n"
            "Desvio estatístico em relação ao histórico do paciente; não é diagnóstico.")


def _claim(alert_id=None):
    """Serialize claims across HTTP tasks and recovery workers/processes."""
    now = time.time()
    with connection() as db:
        with db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT o.alert_id, a.payload FROM notification_outbox o JOIN alerts a ON a.id=o.alert_id "
                "WHERE (o.status='pending' OR (o.status='sending' AND o.lease_until<=?)) "
                "AND (? IS NULL OR o.alert_id=?) ORDER BY a.timestamp, o.alert_id LIMIT 1",
                (now, alert_id, alert_id),
            ).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            db.execute("UPDATE notification_outbox SET status='sending', lease_token=?, lease_until=?, "
                       "attempts=attempts+1 WHERE alert_id=?", (token, now + LEASE_SECONDS, row["alert_id"]))
    return json.loads(row["payload"]), token


def _send(alert):
    url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    if not url:
        return "panel_only", "ALERT_WEBHOOK_URL não configurada; alerta disponível no painel."
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("URL deve ser HTTP(S) e incluir servidor")
        # Include a stable identifier for recipient-side duplicate recognition.
        text = f"ID do alerta: {alert['id']}\n{message(alert)}"
        body = json.dumps({"text": text}).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return "sent", f"HTTP {response.status}"
    except Exception as exc:
        # Do not persist the webhook URL/token potentially embedded in exceptions.
        return "failed", f"{type(exc).__name__}: envio não concluído; confira a configuração e o destino."


def process_one(alert_id=None):
    claimed = _claim(alert_id)
    if claimed is None:
        return False
    alert, token = claimed
    status, detail = _send(alert)
    now = datetime.now(timezone.utc).isoformat()
    with connection() as db:
        with db:
            updated = db.execute(
                "UPDATE notification_outbox SET status=?, lease_token=NULL, lease_until=0 "
                "WHERE alert_id=? AND lease_token=?", (status, alert["id"], token))
            if updated.rowcount:
                db.execute("INSERT INTO notifications(alert_id, status, detail, at) VALUES(?,?,?,?)",
                           (alert["id"], status, detail, now))
    return True


def deliver(alerts: list[dict]) -> None:
    """Fast path after HTTP response; a durable worker recovers interrupted work."""
    for alert in alerts:
        process_one(alert["id"])


@contextmanager
def notification_worker():
    stop = threading.Event()

    def run():
        while not stop.is_set():
            try:
                if process_one():
                    continue
            except Exception:
                # Preserve pending/leased state on DB failures and retry later.
                logger.exception("Falha no processamento da fila de alertas")
            stop.wait(POLL_SECONDS)

    worker = threading.Thread(target=run, name="anomaly-notifications", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=TIMEOUT_SECONDS + 1)


@asynccontextmanager
async def notification_lifespan(app):
    with notification_worker():
        yield
