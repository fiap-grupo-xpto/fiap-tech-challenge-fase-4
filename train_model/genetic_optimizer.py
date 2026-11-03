"""Genetic algorithm optimizer for the tabular lung cancer classifiers.

This script reproduces the GA experiments from the notebook in a reusable,
testable Python module. It trains four sklearn classifiers, runs the three
required experiments, exports the best tabular artifact to the backend, and
stores the optimization results as JSON.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
pd.set_option("mode.string_storage", "python")
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "train_model" / "datasets" / "dataset.csv"
DEFAULT_BACKEND_MODEL_DIR = PROJECT_ROOT / "backend" / "model"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "train_model" / "ga_optimization_results.json"

TARGET_COLUMN = "LUNG_CANCER"
DEMOGRAPHIC_COLUMN = "GENDER"
TARGET_MAPPING = {"NO": 0, "YES": 1}
COLUMN_RENAMES = {
    "YELLOW_FINGERS": "YELLOW FINGERS",
    "PEER_PRESSURE": "PEER PRESSURE",
    "CHRONIC_DISEASE": "CHRONIC DISEASE",
    "ALCOHOL_CONSUMING": "ALCOHOL CONSUMING",
    "SHORTNESS_OF_BREATH": "SHORTNESS OF BREATH",
    "SWALLOWING_DIFFICULTY": "SWALLOWING DIFFICULTY",
    "CHEST_PAIN": "CHEST PAIN",
}

DEFAULT_EXPERIMENTS = (
    {"name": "exp1", "pop_size": 8, "generations": 3, "cx_prob": 0.8, "mut_prob": 0.1, "seed": 1},
    {"name": "exp2", "pop_size": 8, "generations": 3, "cx_prob": 0.8, "mut_prob": 0.3, "seed": 2},
    {"name": "exp3", "pop_size": 12, "generations": 3, "cx_prob": 0.8, "mut_prob": 0.1, "seed": 3},
)

FITNESS_WEIGHTS = {
    "recall": 0.6,
    "specificity": 0.2,
    "f1": 0.2,
    "fairness_penalty": 0.5,
}


@dataclass(frozen=True)
class DatasetBundle:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    groups_val: np.ndarray
    feature_columns: List[str]
    categorical_columns: List[str]
    numeric_columns: List[str]


def prepare_dataset(dataset_path: Path | str = DEFAULT_DATASET_PATH, seed: int = 42) -> DatasetBundle:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path).rename(columns=COLUMN_RENAMES)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].map(TARGET_MAPPING)
    if y.isna().any():
        raise ValueError("Unexpected target labels found in dataset.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y.astype(int),
        test_size=0.4,
        random_state=seed,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp,
    )

    from pandas.api.types import is_numeric_dtype
    numeric_columns = [column for column in X.columns if is_numeric_dtype(X[column])]
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    groups_val = X_val[DEMOGRAPHIC_COLUMN].astype(str).to_numpy() if DEMOGRAPHIC_COLUMN in X_val.columns else np.array(["all"] * len(X_val))

    return DatasetBundle(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        groups_val=groups_val,
        feature_columns=X.columns.tolist(),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )


def build_preprocessor(categorical_columns: Sequence[str], numeric_columns: Sequence[str]) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, list(categorical_columns)),
            ("numeric", numeric_pipeline, list(numeric_columns)),
        ]
    )


def sensitivity(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def specificity(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_group_recall_gap(y_true: Sequence[int], y_pred: Sequence[int], groups: Sequence[str]) -> float:
    group_scores: List[float] = []
    grouped = pd.DataFrame({"y_true": list(y_true), "y_pred": list(y_pred), "group": list(groups)})
    for _, frame in grouped.groupby("group"):
        group_scores.append(recall_score(frame["y_true"], frame["y_pred"], zero_division=0))
    if not group_scores:
        return 0.0
    return float(max(group_scores) - min(group_scores))


def compute_threshold_metrics(y_true: Sequence[int], probabilities: Sequence[float], threshold: float) -> Dict[str, float]:
    y_pred = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def select_best_threshold(y_true: Sequence[int], probabilities: Sequence[float]) -> Dict[str, float]:
    candidates = np.round(np.arange(0.1, 0.91, 0.05), 2)
    scored = [
        compute_threshold_metrics(y_true, probabilities, threshold)
        for threshold in candidates
    ]
    scored.sort(key=lambda item: (item["recall"], item["f1"], item["specificity"]), reverse=True)
    return scored[0]


def make_estimator(model_name: str, params: Dict[str, Any]) -> BaseEstimator:
    params = deepcopy(params)
    if model_name == "RandomForestClassifier":
        params.setdefault("random_state", 42)
        params.setdefault("class_weight", "balanced")
        return RandomForestClassifier(**params)
    if model_name == "ExtraTreesClassifier":
        params.setdefault("random_state", 42)
        params.setdefault("class_weight", "balanced")
        return ExtraTreesClassifier(**params)
    if model_name == "LogisticRegression":
        params.setdefault("random_state", 42)
        params.setdefault("max_iter", 1000)
        return LogisticRegression(**params)
    if model_name == "KNeighborsClassifier":
        return KNeighborsClassifier(**params)
    raise ValueError(f"Unsupported model: {model_name}")


def build_pipeline(
    model_name: str,
    params: Dict[str, Any],
    preprocessor: ColumnTransformer,
) -> Pipeline:
    estimator = make_estimator(model_name, params)
    return Pipeline([("pre", preprocessor), ("clf", estimator)])


MODEL_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "RandomForestClassifier": {
        "n_estimators": {"type": "int", "low": 50, "high": 250},
        "max_depth": {"type": "int", "low": 3, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 10},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 4},
        "max_features": {"type": "float", "low": 0.3, "high": 1.0},
    },
    "LogisticRegression": {
        "C": {"type": "float", "low": 0.01, "high": 10.0},
        "penalty": {"type": "cat", "choices": ["l1", "l2"]},
        "solver": {"type": "cat", "choices": ["liblinear", "saga"]},
        "class_weight": {"type": "cat", "choices": [None, "balanced"]},
        "max_iter": {"type": "int", "low": 200, "high": 1000},
    },
    "KNeighborsClassifier": {
        "n_neighbors": {"type": "int", "low": 3, "high": 15},
        "weights": {"type": "cat", "choices": ["uniform", "distance"]},
        "p": {"type": "int", "low": 1, "high": 2},
    },
    "ExtraTreesClassifier": {
        "n_estimators": {"type": "int", "low": 50, "high": 250},
        "max_depth": {"type": "int", "low": 3, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 10},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 4},
        "max_features": {"type": "float", "low": 0.3, "high": 1.0},
    },
}


def sample_param(spec: Dict[str, Any], rng: random.Random) -> Any:
    if spec["type"] == "int":
        return rng.randint(spec["low"], spec["high"])
    if spec["type"] == "float":
        return round(rng.uniform(spec["low"], spec["high"]), 4)
    if spec["type"] == "cat":
        return rng.choice(spec["choices"])
    raise ValueError(f"Unknown parameter type: {spec['type']}")


def random_individual(param_space: Dict[str, Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    return {name: sample_param(spec, rng) for name, spec in param_space.items()}


def tournament_selection(
    population: Sequence[Dict[str, Any]],
    fitnesses: Sequence[float],
    rng: random.Random,
    k: int = 3,
) -> Dict[str, Any]:
    if len(population) < k:
        k = len(population)
    tournament_indexes = rng.sample(range(len(population)), k)
    winner_index = max(tournament_indexes, key=lambda index: fitnesses[index])
    return deepcopy(population[winner_index])


def crossover(
    parent_a: Dict[str, Any],
    parent_b: Dict[str, Any],
    rng: random.Random,
    cx_prob: float = 0.8,
) -> Dict[str, Any]:
    child = {}
    for key in parent_a:
        child[key] = parent_b[key] if rng.random() < cx_prob else parent_a[key]
    return child


def mutate(
    individual: Dict[str, Any],
    param_space: Dict[str, Dict[str, Any]],
    rng: random.Random,
    mut_prob: float = 0.1,
) -> Dict[str, Any]:
    mutated = deepcopy(individual)
    for key, spec in param_space.items():
        if rng.random() < mut_prob:
            mutated[key] = sample_param(spec, rng)
    return mutated


def evaluate_individual(
    model_name: str,
    indiv: Dict[str, Any],
    dataset: DatasetBundle,
    preprocessor: ColumnTransformer,
) -> Tuple[float, Dict[str, Any]]:
    pipeline = build_pipeline(model_name, indiv, preprocessor)
    try:
        pipeline.fit(dataset.X_train, dataset.y_train)
        predictions = pipeline.predict(dataset.X_val)
    except Exception as exc:
        return -1.0, {"params": deepcopy(indiv), "error": str(exc)}

    recall_value = float(recall_score(dataset.y_val, predictions, zero_division=0))
    specificity_value = float(specificity(dataset.y_val, predictions))
    f1_value = float(f1_score(dataset.y_val, predictions, zero_division=0))
    fairness_penalty = compute_group_recall_gap(dataset.y_val, predictions, dataset.groups_val)

    fitness = (
        FITNESS_WEIGHTS["recall"] * recall_value
        + FITNESS_WEIGHTS["specificity"] * specificity_value
        + FITNESS_WEIGHTS["f1"] * f1_value
        - FITNESS_WEIGHTS["fairness_penalty"] * fairness_penalty
    )
    metrics = {
        "params": deepcopy(indiv),
        "fitness": float(fitness),
        "recall": recall_value,
        "specificity": specificity_value,
        "f1": f1_value,
        "fairness_penalty": float(fairness_penalty),
    }
    return float(fitness), metrics


def run_ga_experiment(
    model_name: str,
    param_space: Dict[str, Dict[str, Any]],
    dataset: DatasetBundle,
    preprocessor: ColumnTransformer,
    pop_size: int,
    generations: int,
    cx_prob: float,
    mut_prob: float,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    population = [random_individual(param_space, rng) for _ in range(pop_size)]
    best_metrics: Dict[str, Any] | None = None
    history: List[float] = []

    for generation in range(generations):
        scored_population = [
            evaluate_individual(model_name, individual, dataset, preprocessor)
            for individual in population
        ]
        fitnesses = [item[0] for item in scored_population]
        metrics_list = [item[1] for item in scored_population]

        current_best = max(metrics_list, key=lambda item: item.get("fitness", -1.0))
        if best_metrics is None or current_best.get("fitness", -1.0) > best_metrics.get("fitness", -1.0):
            best_metrics = deepcopy(current_best)

        history.append(float(best_metrics["fitness"]))
        print(f"Gen {generation + 1}/{generations} - melhor fitness: {best_metrics['fitness']:.4f}")

        elite_index = max(range(len(population)), key=lambda index: fitnesses[index])
        next_population = [deepcopy(population[elite_index])]
        while len(next_population) < pop_size:
            parent_a = tournament_selection(population, fitnesses, rng)
            parent_b = tournament_selection(population, fitnesses, rng)
            child = crossover(parent_a, parent_b, rng, cx_prob=cx_prob)
            child = mutate(child, param_space, rng, mut_prob=mut_prob)
            next_population.append(child)
        population = next_population

    if best_metrics is None:
        raise RuntimeError(f"GA failed to evaluate population for {model_name}.")

    return {
        "best_params": best_metrics["params"],
        "best_fitness": best_metrics["fitness"],
        "validation_metrics": {
            "recall": best_metrics["recall"],
            "specificity": best_metrics["specificity"],
            "f1": best_metrics["f1"],
            "fairness_penalty": best_metrics["fairness_penalty"],
        },
        "history": history,
        "config": {
            "pop_size": pop_size,
            "generations": generations,
            "cx_prob": cx_prob,
            "mut_prob": mut_prob,
            "seed": seed,
        },
    }


def evaluate_on_test_set(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> Dict[str, float]:
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float((predictions == y_test.to_numpy()).mean()),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "specificity": float(specificity(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": float(threshold),
    }


def train_and_export_best_model(
    best_model_name: str,
    best_params: Dict[str, Any],
    dataset: DatasetBundle,
    backend_model_dir: Path | str = DEFAULT_BACKEND_MODEL_DIR,
) -> Dict[str, Any]:
    backend_model_dir = Path(backend_model_dir)
    backend_model_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = build_preprocessor(dataset.categorical_columns, dataset.numeric_columns)
    validation_pipeline = build_pipeline(best_model_name, best_params, preprocessor)
    validation_pipeline.fit(dataset.X_train, dataset.y_train)
    validation_probabilities = validation_pipeline.predict_proba(dataset.X_val)[:, 1]
    best_threshold = select_best_threshold(dataset.y_val, validation_probabilities)

    final_preprocessor = build_preprocessor(dataset.categorical_columns, dataset.numeric_columns)
    final_pipeline = build_pipeline(best_model_name, best_params, final_preprocessor)
    X_train_full = pd.concat([dataset.X_train, dataset.X_val], ignore_index=True)
    y_train_full = pd.concat([dataset.y_train, dataset.y_val], ignore_index=True)
    final_pipeline.fit(X_train_full, y_train_full)

    test_metrics = evaluate_on_test_set(
        final_pipeline,
        dataset.X_test,
        dataset.y_test,
        best_threshold["threshold"],
    )

    artifact = {
        "pipeline": final_pipeline,
        "threshold": best_threshold["threshold"],
        "model_name": best_model_name,
        "feature_columns": dataset.feature_columns,
        "selected_params": best_params,
        "validation_summary": [best_threshold],
        "test_metrics": test_metrics,
        "target_mapping": TARGET_MAPPING,
    }

    artifact_path = backend_model_dir / "lung_cancer_classifier.joblib"
    metadata_path = backend_model_dir / "lung_cancer_classifier.metadata.json"
    joblib.dump(artifact, artifact_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": best_model_name,
                "threshold": best_threshold["threshold"],
                "feature_columns": dataset.feature_columns,
                "selected_params": best_params,
                "validation_summary": [best_threshold],
                "test_metrics": test_metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "threshold_summary": best_threshold,
        "test_metrics": test_metrics,
    }


def summarize_results(results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for model_name, experiment_results in results.items():
        best_experiment_name, best_payload = max(
            experiment_results.items(),
            key=lambda item: item[1]["best_fitness"],
        )
        summary.append(
            {
                "model_name": model_name,
                "best_experiment": best_experiment_name,
                "best_fitness": best_payload["best_fitness"],
                "best_params": best_payload["best_params"],
            }
        )
    summary.sort(key=lambda item: item["best_fitness"], reverse=True)
    return summary


def run_all_experiments(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    backend_model_dir: Path | str = DEFAULT_BACKEND_MODEL_DIR,
    results_path: Path | str = DEFAULT_RESULTS_PATH,
) -> Dict[str, Any]:
    dataset = prepare_dataset(dataset_path)
    preprocessor = build_preprocessor(dataset.categorical_columns, dataset.numeric_columns)

    results: Dict[str, Dict[str, Any]] = {}
    for model_name, param_space in MODEL_SPACES.items():
        print(f"### Executando experimentos para {model_name}")
        results[model_name] = {}
        for config in DEFAULT_EXPERIMENTS:
            experiment_result = run_ga_experiment(
                model_name=model_name,
                param_space=param_space,
                dataset=dataset,
                preprocessor=preprocessor,
                pop_size=config["pop_size"],
                generations=config["generations"],
                cx_prob=config["cx_prob"],
                mut_prob=config["mut_prob"],
                seed=config["seed"],
            )
            results[model_name][config["name"]] = experiment_result

        print(
            "Fim de {model}: exp1={exp1:.4f}, exp2={exp2:.4f}, exp3={exp3:.4f}".format(
                model=model_name,
                exp1=results[model_name]["exp1"]["best_fitness"],
                exp2=results[model_name]["exp2"]["best_fitness"],
                exp3=results[model_name]["exp3"]["best_fitness"],
            )
        )

    ranking = summarize_results(results)
    best_overall = ranking[0]
    export_info = train_and_export_best_model(
        best_model_name=best_overall["model_name"],
        best_params=best_overall["best_params"],
        dataset=dataset,
        backend_model_dir=backend_model_dir,
    )

    payload = {
        "dataset_path": str(Path(dataset_path)),
        "experiments": [dict(item) for item in DEFAULT_EXPERIMENTS],
        "fitness_weights": FITNESS_WEIGHTS,
        "results_by_model": results,
        "ranking": ranking,
        "best_overall": {
            **best_overall,
            "export": export_info,
        },
    }

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GA hyperparameter optimization workflow.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH), help="CSV dataset used in the tabular training workflow.")
    parser.add_argument("--backend-model-dir", default=str(DEFAULT_BACKEND_MODEL_DIR), help="Directory that receives the exported tabular artifact.")
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH), help="JSON file that stores the consolidated GA results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_all_experiments(
        dataset_path=args.dataset_path,
        backend_model_dir=args.backend_model_dir,
        results_path=args.results_path,
    )
    best = payload["best_overall"]
    print("\nResumo final")
    print(f"Melhor modelo: {best['model_name']}")
    print(f"Melhor experimento: {best['best_experiment']}")
    print(f"Fitness: {best['best_fitness']:.4f}")
    print(f"Artefato salvo em: {best['export']['artifact_path']}")
    print(f"Metadados salvos em: {best['export']['metadata_path']}")


if __name__ == "__main__":
    main()
