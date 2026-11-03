import io

from fastapi.testclient import TestClient

import backend.main as backend_main
from train_model.genetic_optimizer import build_pipeline, build_preprocessor, prepare_dataset


def build_test_artifact():
    dataset = prepare_dataset()
    preprocessor = build_preprocessor(dataset.categorical_columns, dataset.numeric_columns)
    pipeline = build_pipeline(
        "KNeighborsClassifier",
        {"n_neighbors": 5, "weights": "distance", "p": 2},
        preprocessor,
    )
    pipeline.fit(dataset.X_train, dataset.y_train)
    return {
        "pipeline": pipeline,
        "threshold": 0.5,
        "model_name": "KNeighborsClassifier",
        "feature_columns": dataset.feature_columns,
        "selected_params": {"n_neighbors": 5, "weights": "distance", "p": 2},
    }, dataset


def test_analyze_tabular_returns_predictions(monkeypatch):
    artifact, dataset = build_test_artifact()
    monkeypatch.setattr(
        backend_main,
        "generate_tabular_interpretation",
        lambda **kwargs: "interpretação simulada",
    )

    sample = dataset.X_test.head(3).rename(
        columns={
            "YELLOW FINGERS": "YELLOW_FINGERS",
            "PEER PRESSURE": "PEER_PRESSURE",
            "CHRONIC DISEASE": "CHRONIC_DISEASE",
            "ALCOHOL CONSUMING": "ALCOHOL_CONSUMING",
            "SHORTNESS OF BREATH": "SHORTNESS_OF_BREATH",
            "SWALLOWING DIFFICULTY": "SWALLOWING_DIFFICULTY",
            "CHEST PAIN": "CHEST_PAIN",
        }
    )

    with TestClient(backend_main.app) as client:
        backend_main.app.state.tabular_artifact = artifact
        backend_main.app.state.tabular_artifact_error = None

        response = client.post(
            "/analyze-tabular",
            files={"csv_file": ("sample.csv", sample.to_csv(index=False).encode("utf-8"), "text/csv")},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["model_name"] == "KNeighborsClassifier"
    assert len(payload["results"]) == 3
    assert payload["results"][0]["llm_interpretation"] == "interpretação simulada"


def test_analyze_tabular_validates_required_columns(monkeypatch):
    artifact, dataset = build_test_artifact()
    monkeypatch.setattr(
        backend_main,
        "generate_tabular_interpretation",
        lambda **kwargs: "interpretação simulada",
    )

    incomplete = dataset.X_test[["GENDER", "AGE", "SMOKING"]].head(2)

    with TestClient(backend_main.app) as client:
        backend_main.app.state.tabular_artifact = artifact
        backend_main.app.state.tabular_artifact_error = None

        response = client.post(
            "/analyze-tabular",
            files={"csv_file": ("sample.csv", incomplete.to_csv(index=False).encode("utf-8"), "text/csv")},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "error"
    assert "missing required columns" in payload["message"].lower()
