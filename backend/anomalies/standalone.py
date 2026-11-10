"""Optional lightweight entrypoint for developing item 3 independently."""
from fastapi import FastAPI
from backend.anomalies.api import router
from backend.anomalies.notifier import notification_lifespan

app = FastAPI(title="Fase 4 — Detecção de anomalias", lifespan=notification_lifespan)
app.include_router(router)
