"""Hyperparameter sweep: 3 topologies x 3 batch sizes = 9 runs.

Each run trains for up to 30 epochs with early stopping on validation AUC,
then scores the validation split at threshold 0.5. Results are written to
``outputs/sweep_results.csv`` sorted by validation ROC-AUC, which is the
metric the final model is chosen on.

    python -m src.sweep --data data/train_neural_network_ready.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import data as data_mod
from src.model import BATCH_SIZES, MAX_EPOCHS, TOPOLOGIES, build_nn_model, early_stopping, set_seed

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"


def score_split(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, precision, sensitivity, specificity, F1 and ROC-AUC."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Validation Accuracy": accuracy_score(y_true, y_pred),
        "Validation Precision": precision_score(y_true, y_pred, zero_division=0),
        "Validation Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Validation Specificity": tn / (tn + fp),
        "Validation F1-score": f1_score(y_true, y_pred, zero_division=0),
        "Validation ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def run_sweep(package: dict, verbose: int = 1) -> pd.DataFrame:
    """Train every (topology, batch size) pair and collect validation metrics."""
    X_train = package["X_train"].values
    X_val = package["X_val"].values
    y_train = package["y_train"]
    y_val = package["y_val"]
    threshold = package["decision_threshold"]

    results = []
    for topology_name, layers in TOPOLOGIES.items():
        for batch_size in BATCH_SIZES:
            experiment_name = f"{topology_name}_Batch_{batch_size}"
            print(f"\nTraining: {experiment_name}", flush=True)

            set_seed()
            model = build_nn_model(input_dim=X_train.shape[1], layers=layers)
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=MAX_EPOCHS,
                batch_size=batch_size,
                callbacks=[early_stopping()],
                class_weight=package["class_weights_dict"],
                verbose=verbose,
            )

            val_prob = model.predict(X_val, verbose=0).ravel()
            results.append(
                {
                    "Experiment": experiment_name,
                    "Topology": topology_name,
                    "Layers": str(layers),
                    "Batch Size": batch_size,
                    **score_split(y_val, val_prob, threshold),
                    "Epochs Trained": len(history.history["loss"]),
                }
            )

    return pd.DataFrame(results).sort_values("Validation ROC-AUC", ascending=False)


def best_epoch(history) -> int:
    """The 1-indexed epoch with the highest validation AUC."""
    import numpy as np

    return int(np.argmax(history.history["val_auc"])) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(data_mod.DEFAULT_CSV))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    package = data_mod.prepare(Path(args.data))
    print(data_mod.describe_splits(package))
    print("Class weights:", package["class_weights_dict"])

    results = run_sweep(package, verbose=args.verbose)
    out_path = output_dir / "sweep_results.csv"
    results.to_csv(out_path, index=False)

    print("\n" + results.to_string(index=False))
    print(f"\nBest experiment: {results.iloc[0]['Experiment']}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
