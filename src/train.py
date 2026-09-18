"""Train the selected model and save everything the later steps need.

The sweep picks the winner on validation ROC-AUC; that was
``Topology_1_Small`` ([64, 32]) at batch size 1024, which early-stopped after
11 epochs with its best validation AUC at epoch 8. This module retrains that
configuration for a fixed number of epochs — no early stopping, since the stop
point is already known — and writes:

  outputs/full_model.keras      the trained network
  outputs/nn_data_package.joblib  splits, feature names, class weights, threshold
  outputs/nn_model_metadata.json  shapes, versions, library versions

    python -m src.train --epochs 8
"""

import argparse
import json
from pathlib import Path

import joblib
import sklearn
import tensorflow as tf

from src import data as data_mod
from src.model import build_nn_model, set_seed

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

BEST_LAYERS = [64, 32]
BEST_BATCH_SIZE = 1024
BEST_EPOCHS = 8  # argmax of validation AUC in the winning sweep run


def train_final_model(
    package: dict,
    layers: list[int] = BEST_LAYERS,
    batch_size: int = BEST_BATCH_SIZE,
    epochs: int = BEST_EPOCHS,
    verbose: int = 1,
):
    """Retrain the winning configuration for a fixed epoch budget."""
    X_train = package["X_train"].values
    X_val = package["X_val"].values

    set_seed()
    model = build_nn_model(input_dim=X_train.shape[1], layers=layers)
    history = model.fit(
        X_train,
        package["y_train"],
        validation_data=(X_val, package["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=package["class_weights_dict"],
        verbose=verbose,
    )
    return model, history


def save_artifacts(model, package: dict, output_dir: Path) -> None:
    """Write the model, the data package and a metadata file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(output_dir / "full_model.keras")

    joblib.dump(
        {
            "X_train_nn": package["X_train"],
            "X_val_nn": package["X_val"],
            "X_test_nn": package["X_test"],
            "y_train": package["y_train"],
            "y_val": package["y_val"],
            "y_test": package["y_test"],
            "feature_names": package["feature_names"],
            "class_weights_dict": package["class_weights_dict"],
            "decision_threshold": package["decision_threshold"],
        },
        output_dir / "nn_data_package.joblib",
    )

    metadata = {
        "model_name": "Credit default risk neural network",
        "model_type": "Dense feed-forward neural network",
        "layers": BEST_LAYERS,
        "batch_size": BEST_BATCH_SIZE,
        "output_activation": "sigmoid",
        "task_type": "binary classification",
        "decision_threshold": package["decision_threshold"],
        "n_input_features": int(package["X_train"].shape[1]),
        "n_train_samples": int(package["X_train"].shape[0]),
        "n_validation_samples": int(package["X_val"].shape[0]),
        "n_test_samples": int(package["X_test"].shape[0]),
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "sklearn_version": sklearn.__version__,
    }
    (output_dir / "nn_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(data_mod.DEFAULT_CSV))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=BEST_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BEST_BATCH_SIZE)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    package = data_mod.prepare(Path(args.data))
    print(data_mod.describe_splits(package))
    print("Class weights:", package["class_weights_dict"])

    model, _ = train_final_model(
        package,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose,
    )

    output_dir = Path(args.output_dir)
    save_artifacts(model, package, output_dir)
    print(f"\nSaved model and data package to {output_dir}")


if __name__ == "__main__":
    main()
