import numpy as np


def calibrate_threshold(scores, correct, target_risk=0.05, min_coverage=0.1):
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_correct = correct[order]
    cumulative_risk = 1.0 - np.cumsum(sorted_correct) / np.arange(1, len(scores) + 1)
    coverage = np.arange(1, len(scores) + 1) / len(scores)
    valid = np.where((cumulative_risk <= target_risk) & (coverage >= min_coverage))[0]
    if len(valid) == 0:
        index = int(np.argmin(cumulative_risk))
    else:
        index = int(valid[-1])
    return float(sorted_scores[index]), float(coverage[index]), float(cumulative_risk[index])


def selective_metrics(scores, correct, threshold):
    accepted = scores >= threshold
    coverage = float(accepted.mean())
    if not accepted.any():
        return {"coverage": 0.0, "selective_risk": 1.0, "accepted_accuracy": 0.0}
    accuracy = float(correct[accepted].mean())
    return {
        "coverage": coverage,
        "selective_risk": 1.0 - accuracy,
        "accepted_accuracy": accuracy,
    }


def ranking_metrics(scores, correct, coverages=(0.1, 0.2, 0.3, 0.5)):
    order = np.argsort(-scores)
    sorted_correct = correct[order]
    cumulative_accuracy = np.cumsum(sorted_correct) / np.arange(1, len(scores) + 1)
    cumulative_risk = 1.0 - cumulative_accuracy
    metrics = {"aurc": float(cumulative_risk.mean())}
    for coverage in coverages:
        count = max(int(round(len(scores) * coverage)), 1)
        metrics[f"accuracy_at_{int(coverage * 100)}pct"] = float(
            sorted_correct[:count].mean()
        )
    return metrics
