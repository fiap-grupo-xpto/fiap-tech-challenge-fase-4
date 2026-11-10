from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from backend.anomalies.demo import demo_observations
from backend.anomalies.notifier import deliver
from backend.anomalies.schemas import ObservationBatch
from backend.anomalies.service import Conflict, acknowledge, dashboard, ingest

router = APIRouter(prefix="/anomalies", tags=["Fase 4 - Anomalias"])


@router.post("/observations")
def submit(batch: ObservationBatch, background: BackgroundTasks):
    try:
        result = ingest(batch.observations)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # Fast path only: ingest has already committed the durable delivery intent.
    # The lifespan worker also recovers pending/expired claims after interruption.
    if result["new_alerts"]:
        background.add_task(deliver, result["new_alerts"])
    return result


@router.get("/patients/{patient_id}")
def get_dashboard(patient_id: str):
    return dashboard(patient_id)


@router.post("/alerts/{alert_id}/acknowledge")
def mark_seen(alert_id: str):
    result = acknowledge(alert_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Alerta não encontrado")
    return result


@router.get("/demo")
def get_demo(patient_id: str = Query(default="DEMO-001", min_length=1, max_length=80, pattern=r"^[\w.-]+$")):
    return {"synthetic": True, "observations": demo_observations(patient_id)}
