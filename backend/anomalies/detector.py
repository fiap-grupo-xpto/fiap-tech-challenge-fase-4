"""Causal robust baseline. Constants are experimental, not clinical limits."""
from statistics import median

from backend.anomalies.schemas import Observation

VERSION = "robust-history-v1"
WINDOW = 12
MIN_HISTORY = 6
MAX_GAP_SECONDS = 15 * 60
MOVEMENT_INTERVAL_SECONDS = 5 * 60
# Real collectors jitter by a few seconds; anything beyond this is a different cadence.
MOVEMENT_INTERVAL_TOLERANCE_SECONDS = 30
SCALE_FLOORS = {"heart_rate": 3, "systolic": 4, "diastolic": 3, "spo2": 0.8,
                "activity_fraction": 0.04}
SCORE_THRESHOLD = 3.5


def candidates(event: Observation, history: list[Observation]) -> tuple[list[dict], str]:
    if event.kind == "prescription":
        if not history:
            return [], "baseline"
        previous = history[-1].prescription
        current = event.prescription
        changes = []
        for name in ("unit", "route", "status"):
            old, new = getattr(previous, name), getattr(current, name)
            if old != new:
                changes.append({"field": name, "previous": old, "current": new})
        # Never compare numerical doses across different units/routes or stopped orders.
        if (previous.unit, previous.route, previous.status) == (current.unit, current.route, "active") and current.status == "active":
            for name in ("dose", "frequency_per_day"):
                old, new = getattr(previous, name), getattr(current, name)
                ratio = abs(new - old) / old
                if ratio >= 0.5:
                    changes.append({"field": name, "previous": old, "current": new,
                                    "relative_change": round(ratio, 4)})
        if not changes:
            return [], "normal"
        return [{"code": "prescription_change", "severity": "review",
                 "message": "Mudança de prescrição requer revisão; pode ser intencional.",
                 "evidence": {"medication": current.medication, "changes": changes,
                              "previous_timestamp": history[-1].timestamp.isoformat()}}], "anomaly"

    # Only a contiguous history in the same context is a comparable baseline.
    recent = []
    next_time = event.timestamp
    for old in reversed(history):
        gap = (next_time - old.timestamp).total_seconds()
        if old.context != event.context or gap > MAX_GAP_SECONDS:
            break
        if event.kind == "movement" and abs(gap - MOVEMENT_INTERVAL_SECONDS) > MOVEMENT_INTERVAL_TOLERANCE_SECONDS:
            break
        recent.append(old)
        next_time = old.timestamp
        if len(recent) == WINDOW:
            break
    if len(recent) < MIN_HISTORY:
        return [], "warming_up"
    findings = []
    for name, value in event.metrics.items():
        values = [item.metrics[name] for item in recent]
        center = median(values)
        mad = median([abs(x - center) for x in values])
        scale = max(1.4826 * mad, SCALE_FLOORS[name])
        score = abs(value - center) / scale
        if score >= SCORE_THRESHOLD:
            findings.append({"code": name + ("_high" if value > center else "_low"),
                             "severity": "review",
                             "message": f"{name}: mudança em relação ao histórico do paciente.",
                             "evidence": {"metric": name, "value": value,
                                          "baseline_median": round(center, 4),
                                          "robust_score": round(score, 4),
                                          "threshold": SCORE_THRESHOLD, "history_count": len(recent),
                                          "context": event.context}})
    return findings, "candidate" if findings else "normal"
