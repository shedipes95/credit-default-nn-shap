"""Loading and splitting the model-ready Home Credit table for the neural network.

The input is a single CSV with one row per loan application: an ``SK_ID_CURR``
identifier, the engineered feature columns, and the binary ``TARGET`` (1 = the
applicant had payment difficulties). Feature engineering happens upstream and is
outside this repo — see ``data/README.md``.

Everything here is deterministic: the split uses ``random_state=42`` and is
stratified on TARGET, so the 8.07% positive rate is preserved in train,
validation and test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_CSV = DATA_DIR / "train_neural_network_ready.csv"

RANDOM_STATE = 42
TEST_FRACTION = 0.2       # held out first, then halved into validation and test
DECISION_THRESHOLD = 0.5


def load_table(csv_path: Path = DEFAULT_CSV) -> pd.DataFrame:
    """Read the model-ready CSV and check the TARGET column is present."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. See data/README.md for how to build it "
            "from the Kaggle Home Credit Default Risk files."
        )
    df = pd.read_csv(csv_path)
    if "TARGET" not in df.columns:
        raise ValueError(f"{csv_path} has no TARGET column.")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop TARGET and the applicant id, return (features, labels)."""
    X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    y = df["TARGET"].astype(int)
    return X, y


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Split 80 / 10 / 10 into train, validation and test, stratified on TARGET."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def class_weights(y_train: pd.Series) -> dict[int, float]:
    """Balanced class weights, so the 8% default class is not drowned out.

    On the full training split this gives roughly {0: 0.54, 1: 6.19}.
    """
    weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    return {0: float(weights[0]), 1: float(weights[1])}


def prepare(csv_path: Path = DEFAULT_CSV) -> dict:
    """Load, split, and compute class weights in one call.

    Returns the splits plus ``feature_names``, ``class_weights_dict`` and the
    decision threshold, which is what the training and evaluation modules expect.
    """
    df = load_table(csv_path)
    X, y = split_features_target(df)
    splits = make_splits(X, y)
    return {
        **splits,
        "feature_names": list(X.columns),
        "class_weights_dict": class_weights(splits["y_train"]),
        "decision_threshold": DECISION_THRESHOLD,
    }


def describe_splits(package: dict) -> str:
    """One-line-per-split summary, handy for logging before a long run."""
    lines = []
    for name in ("train", "val", "test"):
        X = package[f"X_{name}"]
        y = package[f"y_{name}"]
        positive = 100 * float(y.mean())
        lines.append(
            f"{name:<5} rows={X.shape[0]:>7,}  features={X.shape[1]:>4}  "
            f"positive={positive:.2f}%"
        )
    return "\n".join(lines)
