from train_model.genetic_optimizer_image import (
    choose_best_threshold,
    compute_image_fitness,
    sample_param,
)


def test_choose_best_threshold_prefers_high_recall():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.3, 0.8, 0.9]

    threshold, metrics = choose_best_threshold(y_true, y_prob)

    assert 0.1 <= threshold <= 0.9
    assert metrics["recall"] == 1.0
    assert metrics["threshold"] == threshold


def test_compute_image_fitness_uses_all_weighted_metrics():
    fitness = compute_image_fitness(
        {
            "recall": 0.9,
            "specificity": 0.8,
            "f1": 0.85,
            "auc_roc": 0.95,
        }
    )

    assert round(fitness, 4) == round(0.45 * 0.9 + 0.2 * 0.8 + 0.2 * 0.85 + 0.15 * 0.95, 4)


def test_sample_param_supports_log_float():
    value = sample_param({"type": "log_float", "low": 1e-5, "high": 1e-3}, __import__("random").Random(42))

    assert 1e-5 <= value <= 1e-3
