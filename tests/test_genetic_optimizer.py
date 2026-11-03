from pathlib import Path

from train_model.genetic_optimizer import (
    DEFAULT_DATASET_PATH,
    MODEL_SPACES,
    build_pipeline,
    build_preprocessor,
    prepare_dataset,
    run_ga_experiment,
    train_and_export_best_model,
)


def test_prepare_dataset_uses_expected_split_sizes():
    dataset = prepare_dataset(DEFAULT_DATASET_PATH)

    assert len(dataset.X_train) == 1800
    assert len(dataset.X_val) == 600
    assert len(dataset.X_test) == 600
    assert "GENDER" in dataset.feature_columns
    assert "YELLOW FINGERS" in dataset.feature_columns


def test_run_ga_experiment_returns_best_candidate():
    dataset = prepare_dataset(DEFAULT_DATASET_PATH)
    preprocessor = build_preprocessor(dataset.categorical_columns, dataset.numeric_columns)

    result = run_ga_experiment(
        model_name="KNeighborsClassifier",
        param_space=MODEL_SPACES["KNeighborsClassifier"],
        dataset=dataset,
        preprocessor=preprocessor,
        pop_size=4,
        generations=1,
        cx_prob=0.8,
        mut_prob=0.1,
        seed=7,
    )

    assert set(result["best_params"]) == {"n_neighbors", "weights", "p"}
    assert result["best_fitness"] > 0
    assert len(result["history"]) == 1


def test_train_and_export_best_model_writes_artifacts(tmp_path: Path):
    dataset = prepare_dataset(DEFAULT_DATASET_PATH)

    export = train_and_export_best_model(
        best_model_name="KNeighborsClassifier",
        best_params={"n_neighbors": 5, "weights": "distance", "p": 2},
        dataset=dataset,
        backend_model_dir=tmp_path,
    )

    assert Path(export["artifact_path"]).exists()
    assert Path(export["metadata_path"]).exists()
    assert 0.1 <= export["threshold_summary"]["threshold"] <= 0.9
    assert 0 <= export["test_metrics"]["accuracy"] <= 1
