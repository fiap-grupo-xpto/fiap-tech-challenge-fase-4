"""Reproducible synthetic benchmark; not a clinical validation dataset.

Run from repository root: python -m entregas.fase04.src.evaluate_anomalies
"""
import json
import os
import tempfile
from pathlib import Path

from backend.anomalies.demo import demo_observations
from backend.anomalies.schemas import ObservationBatch
from backend.anomalies.service import ingest


def evaluate():
    results = {}
    previous_path = os.environ.get("ANOMALY_DB_PATH")
    try:
        with tempfile.TemporaryDirectory() as directory:
            os.environ["ANOMALY_DB_PATH"] = str(Path(directory) / "evaluation.db")
            for scenario in ("normal", "strong_change", "subtle_change"):
                counts = {kind: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for kind in ("vitals", "movement", "prescription")}
                for seed in range(1000, 1020):
                    raw = demo_observations(f"EVAL-{scenario}-{seed}", seed, anomalous=scenario != "normal")
                    if scenario == "subtle_change":
                        # Smaller injected changes test the detector's deliberate blind spots.
                        for e in raw:
                            index = int(e["event_id"].rsplit("-", 1)[1])
                            if e["kind"] == "prescription" and index >= 12:
                                e["prescription"]["dose"] = 12
                            elif 12 <= index <= 15:
                                if e["kind"] == "movement":
                                    e["metrics"]["activity_fraction"] += 0.39
                                else:
                                    for metric, reduction in (("heart_rate", 30), ("systolic", 24), ("diastolic", 18)):
                                        e["metrics"][metric] -= reduction
                                    e["metrics"]["spo2"] += 6
                    batch = ObservationBatch(observations=raw)
                    output = ingest(batch.observations)
                    for event, result in zip(raw, output["results"]):
                        index = int(event["event_id"].rsplit("-", 1)[1])
                        if result["state"] in ("warming_up", "baseline"):
                            continue
                        truth = scenario != "normal" and (index == 12 if event["kind"] == "prescription" else 12 <= index <= 15)
                        predicted = result["state"] == "anomaly"
                        key = "tp" if truth and predicted else "fn" if truth else "fp" if predicted else "tn"
                        counts[event["kind"]][key] += 1
                for value in counts.values():
                    tp, fp, tn, fn = (value[k] for k in ("tp", "fp", "tn", "fn"))
                    value.update(precision=round(tp / (tp + fp), 4) if tp + fp else None,
                                 recall=round(tp / (tp + fn), 4) if tp + fn else None,
                                 f1=round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else None,
                                 accuracy=round((tp + tn) / (tp + tn + fp + fn), 4),
                                 always_normal_accuracy=round((tn + fp) / (tp + tn + fp + fn), 4))
                results[scenario] = counts
    finally:
        if previous_path is None:
            os.environ.pop("ANOMALY_DB_PATH", None)
        else:
            os.environ["ANOMALY_DB_PATH"] = previous_path
    return {"dataset": "synthetic-only", "seeds": [1000, 1019], "patients": 60,
            "method": "robust-history-v1", "unit": "observation per domain after warm-up",
            "labeling": "Known injected intervals, including the first changed sample; not detector-derived labels.",
            "limitations": "Synthetic benchmark, no clinical validity. Thresholds fixed before evaluation; no fitted model. Seeds differ from UI demo (42).",
            "results": results}


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "results" / "anomalies_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evaluate(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
