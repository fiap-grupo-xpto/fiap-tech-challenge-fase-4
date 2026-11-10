"""Synthetic fixture, no real patients or real medication instructions."""
import random
from datetime import datetime, timedelta, timezone


def demo_observations(patient_id="DEMO-001", seed=42, anomalous=True):
    rng = random.Random(seed)
    events = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(24):
        timestamp = (start + timedelta(minutes=5 * i)).isoformat()
        changed = anomalous and 12 <= i <= 15
        vitals = {"heart_rate": 75 + rng.uniform(-3, 3), "systolic": 120 + rng.uniform(-4, 4),
                  "diastolic": 80 + rng.uniform(-3, 3), "spo2": 97 + rng.uniform(-0.5, 0.5)}
        if changed:
            vitals["heart_rate"] += 35
            vitals["spo2"] -= 7
            vitals["systolic"] += 30
            vitals["diastolic"] += 22
        for kind, metrics in (("vitals", vitals), ("movement", {"activity_fraction": (0.08 if changed else 0.55) + rng.uniform(-0.02, 0.02)})):
            events.append({"event_id": f"{patient_id}-{kind}-{i}", "patient_id": patient_id,
                           "timestamp": timestamp, "kind": kind, "metrics": metrics})
        if i in (0, 12, 20):
            events.append({"event_id": f"{patient_id}-prescription-{i}", "patient_id": patient_id,
                           "timestamp": timestamp, "kind": "prescription",
                           "prescription": {"medication": "MEDICAMENTO-FICTICIO", "dose": 20 if anomalous and i >= 12 else 10,
                                            "unit": "mg", "frequency_per_day": 1, "route": "simulada"}})
    return events
