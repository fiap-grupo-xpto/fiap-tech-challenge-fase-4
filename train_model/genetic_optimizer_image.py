"""Genetic algorithm optimizer for the image classification workflow.

This module mirrors the image notebook in a reusable script and applies
genetic search over key training hyperparameters of the CNN pipeline.
The best model is exported to `backend/model/best.keras` so the API can use
the optimized image artifact productively.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
pd.set_option("mode.string_storage", "python")
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "train_model" / "datasets"
DATASET_ZIP_PATH = DATASETS_DIR / "The IQ-OTHNCCD lung cancer dataset.zip"
DATASET_EXTRACTED_DIR = DATASETS_DIR / "extracted"
DATASET_ROOT_DIR = DATASET_EXTRACTED_DIR / "The IQ-OTHNCCD lung cancer dataset"
DEFAULT_BACKEND_MODEL_DIR = PROJECT_ROOT / "backend" / "model"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "train_model" / "ga_image_optimization_results.json"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "train_model" / "ga_image_runs"

CLASS_INFO = {
    "Normal cases": {"label": 0, "display": "Normal"},
    "Bengin cases": {"label": 0, "display": "Benigno"},
    "Malignant cases": {"label": 1, "display": "Maligno"},
}

LABEL_NAMES = {0: "nao_maligno", 1: "maligno"}

IMAGE_EXPERIMENTS = (
    {"name": "exp1", "pop_size": 2, "generations": 1, "cx_prob": 0.8, "mut_prob": 0.15, "seed": 11},
    {"name": "exp2", "pop_size": 2, "generations": 1, "cx_prob": 0.8, "mut_prob": 0.30, "seed": 22},
    {"name": "exp3", "pop_size": 3, "generations": 2, "cx_prob": 0.8, "mut_prob": 0.20, "seed": 33},
)

FITNESS_WEIGHTS = {
    "recall": 0.45,
    "specificity": 0.20,
    "f1": 0.20,
    "auc_roc": 0.15,
}

IMAGE_PARAM_SPACE: Dict[str, Dict[str, Any]] = {
    "learning_rate": {"type": "log_float", "low": 5e-5, "high": 5e-4},
    "batch_size": {"type": "cat", "choices": [8, 16]},
    "frozen_epochs": {"type": "int", "low": 2, "high": 4},
    "fine_tune_epochs": {"type": "int", "low": 1, "high": 2},
    "dense_units": {"type": "cat", "choices": [64, 128, 256]},
    "dropout_rate": {"type": "float", "low": 0.20, "high": 0.50},
    "use_class_weight": {"type": "cat", "choices": [True, False]},
    "unfreeze_layers": {"type": "cat", "choices": [10, 20, 40]},
}


def get_tf_modules():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - depends on local runtime
        raise RuntimeError(
            "TensorFlow nao esta instalado no interpretador atual. "
            "Execute este script com Python 3.11 + tensorflow."
        ) from exc
    return tf, keras


@dataclass(frozen=True)
class ImageDatasetBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    dataset_dir: Path


def set_seed(seed: int = 42) -> None:
    tf, _ = get_tf_modules()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_image_dataset() -> Path:
    if DATASET_ROOT_DIR.exists():
        return DATASET_ROOT_DIR

    if not DATASET_ZIP_PATH.exists():
        raise FileNotFoundError(f"Arquivo zip do dataset de imagem nao encontrado: {DATASET_ZIP_PATH}")

    DATASET_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    with ZipFile(DATASET_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATASET_EXTRACTED_DIR)

    if not DATASET_ROOT_DIR.exists():
        raise FileNotFoundError(f"Diretorio extraido do dataset nao encontrado: {DATASET_ROOT_DIR}")
    return DATASET_ROOT_DIR


def collect_image_records(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for folder_name, info in CLASS_INFO.items():
        folder = dataset_dir / folder_name
        if not folder.exists():
            continue
        for image_path in sorted(folder.glob("*.jpg")):
            rows.append(
                {
                    "image_path": str(image_path),
                    "label": int(info["label"]),
                    "class_original": folder_name,
                }
            )

    if not rows:
        raise RuntimeError(f"Nenhuma imagem encontrada em {dataset_dir}")
    return pd.DataFrame(rows)


def split_records(
    records: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_fraction = 1.0 - val_fraction - test_fraction
    if train_fraction <= 0:
        raise ValueError("Fractions invalidas para split do dataset.")

    train_df, temp_df = train_test_split(
        records,
        test_size=(1.0 - train_fraction),
        random_state=seed,
        stratify=records["label"],
    )
    relative_test_fraction = test_fraction / (val_fraction + test_fraction)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_fraction,
        random_state=seed,
        stratify=temp_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_numpy_dataset(records: pd.DataFrame, image_size: int = 224) -> Tuple[np.ndarray, np.ndarray]:
    _, keras = get_tf_modules()
    images: List[np.ndarray] = []
    labels: List[float] = []
    for row in records.itertuples(index=False):
        img = keras.utils.load_img(row.image_path, target_size=(image_size, image_size))
        img_array = keras.utils.img_to_array(img)
        images.append(img_array)
        labels.append(float(row.label))
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def prepare_image_dataset(image_size: int = 224, seed: int = 42) -> ImageDatasetBundle:
    dataset_dir = ensure_image_dataset()
    records = collect_image_records(dataset_dir)
    train_df, val_df, test_df = split_records(records, seed=seed)
    x_train, y_train = build_numpy_dataset(train_df, image_size=image_size)
    x_val, y_val = build_numpy_dataset(val_df, image_size=image_size)
    x_test, y_test = build_numpy_dataset(test_df, image_size=image_size)
    return ImageDatasetBundle(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        dataset_dir=dataset_dir,
    )


def calculate_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    y_train = np.asarray(y_train).astype(np.int32)
    total = len(y_train)
    negatives = max(int(np.sum(y_train == 0)), 1)
    positives = max(int(np.sum(y_train == 1)), 1)
    return {
        0: total / (2.0 * negatives),
        1: total / (2.0 * positives),
    }


def binary_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float32)
    y_pred = (y_prob >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    f1 = float((2 * precision * recall) / max(precision + recall, 1e-8))
    accuracy = float((tp + tn) / max(len(y_true), 1))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": float(threshold),
    }


def choose_best_threshold(y_true: Sequence[int], y_prob: Sequence[float]) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics = None
    best_score = -1.0
    for threshold in np.linspace(0.1, 0.9, 81):
        metrics = binary_metrics(y_true, y_prob, threshold=float(threshold))
        score = metrics["recall"] * 2.0 + metrics["specificity"] + metrics["f1"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics or binary_metrics(y_true, y_prob, 0.5)


def evaluate_probabilities(y_true: Sequence[int], y_prob: Sequence[float], threshold: float) -> Dict[str, float]:
    metrics = binary_metrics(y_true, y_prob, threshold=threshold)
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float32)
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = None
    try:
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["average_precision"] = None
    return metrics


def build_image_model(
    image_size: int,
    learning_rate: float,
    dense_units: int,
    dropout_rate: float,
    fine_tune: bool,
):
    _, keras = get_tf_modules()

    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.03),
            keras.layers.RandomZoom(0.08),
            keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    weights = "imagenet"
    try:
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=(image_size, image_size, 3),
        )
    except Exception:
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(image_size, image_size, 3),
        )

    base_model.trainable = fine_tune

    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = augmentation(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate / 2.0)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lung_cancer_effnetb0")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model, base_model


def create_callbacks(checkpoint_path: Path):
    _, keras = get_tf_modules()
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=1,
            restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=0,
        ),
    ]


def compute_image_fitness(test_metrics: Dict[str, Any]) -> float:
    auc_value = float(test_metrics.get("auc_roc") or 0.0)
    return (
        FITNESS_WEIGHTS["recall"] * float(test_metrics["recall"])
        + FITNESS_WEIGHTS["specificity"] * float(test_metrics["specificity"])
        + FITNESS_WEIGHTS["f1"] * float(test_metrics["f1"])
        + FITNESS_WEIGHTS["auc_roc"] * auc_value
    )


def train_image_candidate(
    dataset: ImageDatasetBundle,
    params: Dict[str, Any],
    output_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    tf, keras = get_tf_modules()
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.keras"
    info_path = output_dir / "best_info.json"
    metrics_path = output_dir / "metrics.json"

    class_weight = calculate_class_weights(dataset.y_train) if params["use_class_weight"] else None

    model, base_model = build_image_model(
        image_size=224,
        learning_rate=float(params["learning_rate"]),
        dense_units=int(params["dense_units"]),
        dropout_rate=float(params["dropout_rate"]),
        fine_tune=False,
    )
    callbacks = create_callbacks(checkpoint_path)

    history_initial = model.fit(
        dataset.x_train,
        dataset.y_train,
        validation_data=(dataset.x_val, dataset.y_val),
        epochs=int(params["frozen_epochs"]),
        batch_size=int(params["batch_size"]),
        callbacks=callbacks,
        verbose=0,
        class_weight=class_weight,
    )
    histories = [history_initial.history]

    if int(params["fine_tune_epochs"]) > 0:
        base_model.trainable = True
        unfreeze_layers = int(params["unfreeze_layers"])
        if len(base_model.layers) > unfreeze_layers:
            for layer in base_model.layers[:-unfreeze_layers]:
                layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=float(params["learning_rate"]) / 10.0),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.AUC(name="auc"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )
        history_finetune = model.fit(
            dataset.x_train,
            dataset.y_train,
            validation_data=(dataset.x_val, dataset.y_val),
            epochs=int(params["fine_tune_epochs"]),
            batch_size=int(params["batch_size"]),
            callbacks=callbacks,
            verbose=0,
            class_weight=class_weight,
        )
        histories.append(history_finetune.history)

    best_model = keras.models.load_model(checkpoint_path)
    y_prob_val = best_model.predict(dataset.x_val, verbose=0).ravel()
    best_threshold, val_metrics = choose_best_threshold(dataset.y_val, y_prob_val)
    y_prob_test = best_model.predict(dataset.x_test, verbose=0).ravel()
    test_metrics = evaluate_probabilities(dataset.y_test, y_prob_test, threshold=best_threshold)
    fitness = compute_image_fitness(test_metrics)

    history_total: Dict[str, List[float]] = {}
    for hist in histories:
        for key, values in hist.items():
            history_total.setdefault(key, []).extend([float(value) for value in values])

    metadata = {
        "image_size": 224,
        "class_names": LABEL_NAMES,
        "positive_class": "maligno",
        "negative_class": "nao_maligno",
        "dataset_dir": str(dataset.dataset_dir),
        "best_threshold": float(best_threshold),
        "class_weight": class_weight,
        "backbone": "EfficientNetB0",
        "selected_params": params,
    }
    info_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "history": history_total,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "fitness": fitness,
                "train_images": len(dataset.train_df),
                "val_images": len(dataset.val_df),
                "test_images": len(dataset.test_df),
                **metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    keras.backend.clear_session()
    tf.keras.backend.clear_session()

    return {
        "params": deepcopy(params),
        "fitness": float(fitness),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_threshold": float(best_threshold),
        "checkpoint_path": str(checkpoint_path),
        "info_path": str(info_path),
        "metrics_path": str(metrics_path),
    }


def sample_param(spec: Dict[str, Any], rng: random.Random) -> Any:
    if spec["type"] == "int":
        return rng.randint(spec["low"], spec["high"])
    if spec["type"] == "float":
        return round(rng.uniform(spec["low"], spec["high"]), 4)
    if spec["type"] == "log_float":
        low = math.log10(spec["low"])
        high = math.log10(spec["high"])
        return round(10 ** rng.uniform(low, high), 6)
    if spec["type"] == "cat":
        return rng.choice(spec["choices"])
    raise ValueError(f"Tipo de parametro desconhecido: {spec['type']}")


def random_individual(param_space: Dict[str, Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    return {name: sample_param(spec, rng) for name, spec in param_space.items()}


def tournament_selection(
    population: Sequence[Dict[str, Any]],
    fitnesses: Sequence[float],
    rng: random.Random,
    k: int = 2,
) -> Dict[str, Any]:
    k = min(k, len(population))
    indexes = rng.sample(range(len(population)), k)
    winner = max(indexes, key=lambda index: fitnesses[index])
    return deepcopy(population[winner])


def crossover(parent_a: Dict[str, Any], parent_b: Dict[str, Any], rng: random.Random, cx_prob: float) -> Dict[str, Any]:
    child = {}
    for key in parent_a:
        child[key] = parent_b[key] if rng.random() < cx_prob else parent_a[key]
    return child


def mutate(
    individual: Dict[str, Any],
    param_space: Dict[str, Dict[str, Any]],
    rng: random.Random,
    mut_prob: float,
) -> Dict[str, Any]:
    mutated = deepcopy(individual)
    for key, spec in param_space.items():
        if rng.random() < mut_prob:
            mutated[key] = sample_param(spec, rng)
    return mutated


def run_image_ga_experiment(
    dataset: ImageDatasetBundle,
    param_space: Dict[str, Dict[str, Any]],
    pop_size: int,
    generations: int,
    cx_prob: float,
    mut_prob: float,
    seed: int,
    runs_root: Path,
    experiment_name: str,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    population = [random_individual(param_space, rng) for _ in range(pop_size)]
    best_payload: Dict[str, Any] | None = None
    history: List[float] = []
    candidate_index = 0

    for generation in range(generations):
        scored_population = []
        for individual in population:
            candidate_dir = runs_root / experiment_name / f"gen_{generation + 1}_cand_{candidate_index + 1}"
            result = train_image_candidate(dataset, individual, candidate_dir, seed=seed + candidate_index)
            scored_population.append(result)
            candidate_index += 1

        fitnesses = [item["fitness"] for item in scored_population]
        current_best = max(scored_population, key=lambda item: item["fitness"])
        if best_payload is None or current_best["fitness"] > best_payload["fitness"]:
            best_payload = deepcopy(current_best)

        history.append(float(best_payload["fitness"]))
        print(f"Gen {generation + 1}/{generations} - melhor fitness: {best_payload['fitness']:.4f}")

        elite_idx = max(range(len(population)), key=lambda index: fitnesses[index])
        next_population = [deepcopy(population[elite_idx])]
        while len(next_population) < pop_size:
            parent_a = tournament_selection(population, fitnesses, rng)
            parent_b = tournament_selection(population, fitnesses, rng)
            child = crossover(parent_a, parent_b, rng, cx_prob=cx_prob)
            child = mutate(child, param_space, rng, mut_prob=mut_prob)
            next_population.append(child)
        population = next_population

    if best_payload is None:
        raise RuntimeError("O AG de imagem nao conseguiu gerar nenhum candidato valido.")

    return {
        "best_params": best_payload["params"],
        "best_fitness": best_payload["fitness"],
        "validation_metrics": best_payload["validation_metrics"],
        "test_metrics": best_payload["test_metrics"],
        "best_threshold": best_payload["best_threshold"],
        "history": history,
        "artifacts": {
            "checkpoint_path": best_payload["checkpoint_path"],
            "info_path": best_payload["info_path"],
            "metrics_path": best_payload["metrics_path"],
        },
        "config": {
            "pop_size": pop_size,
            "generations": generations,
            "cx_prob": cx_prob,
            "mut_prob": mut_prob,
            "seed": seed,
        },
    }


def export_best_image_artifact(best_result: Dict[str, Any], backend_model_dir: Path, dataset: ImageDatasetBundle) -> Dict[str, Any]:
    backend_model_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = Path(best_result["artifacts"]["checkpoint_path"])
    best_info = Path(best_result["artifacts"]["info_path"])
    best_metrics = Path(best_result["artifacts"]["metrics_path"])

    exported_model = backend_model_dir / "best.keras"
    exported_info = backend_model_dir / "best_info.json"
    exported_metrics = backend_model_dir / "metrics.json"

    shutil.copy2(best_checkpoint, exported_model)
    shutil.copy2(best_info, exported_info)
    shutil.copy2(best_metrics, exported_metrics)

    payload = json.loads(exported_metrics.read_text(encoding="utf-8"))
    payload["dataset_dir"] = str(dataset.dataset_dir)
    exported_metrics.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "model_path": str(exported_model),
        "info_path": str(exported_info),
        "metrics_path": str(exported_metrics),
    }


def run_all_image_experiments(
    backend_model_dir: Path | str = DEFAULT_BACKEND_MODEL_DIR,
    results_path: Path | str = DEFAULT_RESULTS_PATH,
    runs_root: Path | str = DEFAULT_RUNS_DIR,
) -> Dict[str, Any]:
    dataset = prepare_image_dataset(image_size=224, seed=42)
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    for config in IMAGE_EXPERIMENTS:
        print(f"### Executando experimento de imagem {config['name']}")
        results[config["name"]] = run_image_ga_experiment(
            dataset=dataset,
            param_space=IMAGE_PARAM_SPACE,
            pop_size=config["pop_size"],
            generations=config["generations"],
            cx_prob=config["cx_prob"],
            mut_prob=config["mut_prob"],
            seed=config["seed"],
            runs_root=runs_root,
            experiment_name=config["name"],
        )

    ranking = sorted(
        (
            {
                "experiment_name": name,
                "best_fitness": payload["best_fitness"],
                "best_params": payload["best_params"],
                "test_metrics": payload["test_metrics"],
                "best_threshold": payload["best_threshold"],
            }
            for name, payload in results.items()
        ),
        key=lambda item: item["best_fitness"],
        reverse=True,
    )
    best_overall = ranking[0]
    export_info = export_best_image_artifact(
        best_result=results[best_overall["experiment_name"]],
        backend_model_dir=Path(backend_model_dir),
        dataset=dataset,
    )

    payload = {
        "dataset_dir": str(dataset.dataset_dir),
        "experiments": [dict(item) for item in IMAGE_EXPERIMENTS],
        "fitness_weights": FITNESS_WEIGHTS,
        "param_space": IMAGE_PARAM_SPACE,
        "results_by_experiment": results,
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
    parser = argparse.ArgumentParser(description="Run the GA optimization workflow for image classification.")
    parser.add_argument("--backend-model-dir", default=str(DEFAULT_BACKEND_MODEL_DIR))
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_all_image_experiments(
        backend_model_dir=args.backend_model_dir,
        results_path=args.results_path,
        runs_root=args.runs_root,
    )
    best = payload["best_overall"]
    print("\nResumo final do AG de imagem")
    print(f"Melhor experimento: {best['experiment_name']}")
    print(f"Fitness: {best['best_fitness']:.4f}")
    print(f"Modelo exportado em: {best['export']['model_path']}")
    print(f"Metricas exportadas em: {best['export']['metrics_path']}")


if __name__ == "__main__":
    main()
