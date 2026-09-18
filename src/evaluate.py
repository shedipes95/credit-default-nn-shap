"""Evaluate the trained network on the held-out test set.

Accuracy is reported but is not the point: on an 8%-positive problem the
interesting numbers are recall (how many real defaults were caught), precision
(how many flagged applicants actually defaulted), ROC-AUC (ranking quality) and
average precision (the PR-curve summary, which is the honest one under class
imbalance).

Writes a metrics JSON plus four figures: confusion matrix, ROC curve,
precision-recall curve, and the predicted-probability distribution split by
true label.

    python -m src.evaluate
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"


def load_model_and_data(output_dir: Path) -> tuple:
    """Load the saved model and data package written by ``src.train``."""
    model = tf.keras.models.load_model(output_dir / "full_model.keras", compile=False)
    package = joblib.load(output_dir / "nn_data_package.joblib")

    X_test = package["X_test_nn"]
    if X_test.shape[1] != model.input_shape[-1]:
        raise ValueError(
            f"Model expects {model.input_shape[-1]} features, "
            f"test set has {X_test.shape[1]}."
        )
    return model, package


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Headline metrics plus the raw confusion-matrix counts."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def plot_confusion_matrix(cm: np.ndarray, output_dir: Path) -> Path:
    """Counts, not proportions — the raw false-positive count is the story."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion matrix - neural network")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.xticks([0, 1], ["Non-default", "Default"])
    plt.yticks([0, 1], ["Non-default", "Default"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_roc(y_true, y_prob, roc_auc: float, output_dir: Path) -> Path:
    """ROC curve against the diagonal."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve - neural network")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "roc_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_precision_recall(y_true, y_prob, output_dir: Path) -> Path:
    """Precision-recall curve, the more honest view under 8% positives."""
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall_vals, precision_vals, label=f"Average precision = {avg_precision:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve - neural network")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "precision_recall_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_probability_distribution(y_true, y_prob, threshold: float, output_dir: Path) -> Path:
    """Predicted probabilities for each true class, with the threshold marked."""
    y_true = np.asarray(y_true)
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="Actual non-default")
    plt.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="Actual default")
    plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold}")
    plt.xlabel("Predicted probability of default")
    plt.ylabel("Number of applicants")
    plt.title("Distribution of predicted default probabilities")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "probability_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def evaluate(output_dir: Path, make_plots: bool = True) -> dict:
    """Score the test set, optionally write the figures, return the metrics."""
    model, package = load_model_and_data(output_dir)
    X_test = package["X_test_nn"]
    y_test = package["y_test"]
    threshold = package["decision_threshold"]

    y_prob = model.predict(X_test, verbose=0).ravel()
    metrics = compute_metrics(y_test, y_prob, threshold)

    if make_plots:
        cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        plot_confusion_matrix(cm, output_dir)
        plot_roc(y_test, y_prob, metrics["roc_auc"], output_dir)
        plot_precision_recall(y_test, y_prob, output_dir)
        plot_probability_distribution(y_test, y_prob, threshold, output_dir)

    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "nn_test_predictions.npz",
        y_prob=y_prob,
        y_pred=(y_prob >= threshold).astype(int),
    )

    print(classification_report(y_test, (y_prob >= threshold).astype(int), zero_division=0))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics = evaluate(output_dir, make_plots=not args.no_plots)

    for key in (
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "roc_auc",
        "average_precision",
    ):
        print(f"{key:<20} {metrics[key]:.4f}")
    print(
        f"confusion matrix     tn={metrics['tn']} fp={metrics['fp']} "
        f"fn={metrics['fn']} tp={metrics['tp']}"
    )


if __name__ == "__main__":
    main()
