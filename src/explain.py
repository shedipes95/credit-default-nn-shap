"""SHAP explanations for the trained network - global and per-case.

Kernel SHAP is model-agnostic, which is what you need for a black-box network,
but it is expensive: 499 features times every test row is not affordable. So a
small background sample (25 training rows) and a small explanation sample
(40 test rows) are drawn with a fixed seed, ``nsamples=100`` and
``l1_reg="num_features(20)"``. That is a real limitation and the README says so.

Two outputs:

  global  - mean absolute SHAP per feature, ranked, plus a bar chart and a
            summary-style scatter of the top 20
  local   - the most confident true positive, true negative, false positive and
            false negative in the explanation sample, each with its top-10
            feature contributions

    python -m src.explain --background 25 --explain 40
"""

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

BACKGROUND_SIZE = 25
EXPLAIN_SIZE = 40
NSAMPLES = 100
L1_REG = "num_features(20)"
SEED = 42


def sample_rows(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    background_size: int = BACKGROUND_SIZE,
    explain_size: int = EXPLAIN_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Draw the background and explanation samples with a fixed seed."""
    rng = np.random.RandomState(seed)
    background_idx = rng.choice(X_train.shape[0], size=background_size, replace=False)
    explain_idx = rng.choice(X_test.shape[0], size=explain_size, replace=False)
    return X_train.iloc[background_idx], X_test.iloc[explain_idx], background_idx, explain_idx


def make_predict_fn(model, feature_names: list[str]):
    """Wrap the Keras model so SHAP gets a plain array-in, array-out callable."""

    def model_predict(data: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(data, columns=feature_names)
        return model.predict(frame, batch_size=128, verbose=0).ravel()

    return model_predict


def compute_shap_values(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    feature_names: list[str],
    nsamples: int = NSAMPLES,
    l1_reg: str = L1_REG,
) -> np.ndarray:
    """Run Kernel SHAP and return a (rows, features) array.

    Kernel SHAP warns about a singular regression here: 499 features against a
    40-row sample is under-determined, so it falls back to the Moore-Penrose
    pseudoinverse. The L1 term keeps it tractable. The values are approximate.
    """
    explainer = shap.KernelExplainer(make_predict_fn(model, feature_names), X_background)
    values = np.array(explainer.shap_values(X_explain, nsamples=nsamples, l1_reg=l1_reg))
    if values.ndim == 3:  # some SHAP versions add a trailing output axis
        values = values[:, :, 0]
    if values.shape[1] != len(feature_names):
        raise ValueError("SHAP values do not line up with the feature names.")
    return values


def global_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP per feature, highest first."""
    return pd.DataFrame(
        {"Feature": feature_names, "Mean_ABS_SHAP": np.abs(shap_values).mean(axis=0)}
    ).sort_values("Mean_ABS_SHAP", ascending=False)


def plot_global_importance(importance: pd.DataFrame, output_dir: Path, top_n: int = 20) -> Path:
    """Horizontal bar chart of the top features by mean absolute SHAP."""
    top = importance.head(top_n)
    plt.figure(figsize=(10, 8))
    plt.barh(top["Feature"][::-1], top["Mean_ABS_SHAP"][::-1])
    plt.xlabel("Mean absolute SHAP value")
    plt.title(f"Top {top_n} global SHAP feature importance")
    plt.tight_layout()
    path = output_dir / "shap_global_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_summary_scatter(
    shap_values: np.ndarray,
    importance: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
    top_n: int = 20,
) -> Path:
    """Summary-style scatter drawn by hand.

    ``shap.summary_plot`` fails on a colorbar in some headless environments, so
    this draws the same idea directly: one row per feature, one point per
    observation, jittered vertically so the points do not sit on top of each
    other.
    """
    top_features = importance.head(top_n)["Feature"].tolist()
    indices = [feature_names.index(f) for f in top_features]
    rng = np.random.RandomState(SEED)

    plt.figure(figsize=(10, 8))
    for position, feature_idx in enumerate(indices):
        values = shap_values[:, feature_idx]
        jitter = rng.normal(0, 0.08, size=len(values))
        plt.scatter(values, np.full(len(values), position) + jitter, alpha=0.7, s=25)

    plt.yticks(range(len(top_features)), top_features)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("SHAP value")
    plt.ylabel("Feature")
    plt.title("SHAP summary - neural network")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    path = output_dir / "shap_summary.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def prediction_table(
    model, X_explain: pd.DataFrame, y_true: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Actual label, predicted label and default probability per explained row."""
    probs = model.predict(X_explain, batch_size=128, verbose=0).ravel()
    return pd.DataFrame(
        {
            "Actual": np.asarray(y_true),
            "Predicted": (probs >= threshold).astype(int),
            "Default_Probability": probs,
        }
    )


def select_cases(predictions: pd.DataFrame) -> dict[str, int]:
    """Pick the most confident TP, TN, FP and FN in the explanation sample.

    "Most confident" means highest default probability for the cases the model
    called a default, and lowest for the cases it called safe. A category with
    no rows is simply left out.
    """
    actual, predicted = predictions["Actual"], predictions["Predicted"]
    groups = {
        "true_positive": (predictions[(actual == 1) & (predicted == 1)], False),
        "true_negative": (predictions[(actual == 0) & (predicted == 0)], True),
        "false_positive": (predictions[(actual == 0) & (predicted == 1)], False),
        "false_negative": (predictions[(actual == 1) & (predicted == 0)], True),
    }
    selected = {}
    for name, (frame, ascending) in groups.items():
        if len(frame):
            selected[name] = int(
                frame.sort_values("Default_Probability", ascending=ascending).index[0]
            )
    return selected


def explain_case(
    case_name: str,
    row_index: int,
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    feature_names: list[str],
    predictions: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Top-``top_n`` feature contributions for one row, by absolute SHAP value."""
    case = pd.DataFrame(
        {
            "Feature": feature_names,
            "Feature_Value": X_explain.iloc[row_index].values,
            "SHAP_Value": shap_values[row_index],
        }
    )
    case["ABS_SHAP"] = case["SHAP_Value"].abs()
    case = case.sort_values("ABS_SHAP", ascending=False).head(top_n)

    row = predictions.loc[row_index]
    print("=" * 80)
    print(case_name)
    print(f"Actual: {int(row['Actual'])}  Predicted: {int(row['Predicted'])}  "
          f"P(default): {row['Default_Probability']:.4f}")
    print(case.to_string(index=False))
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--background", type=int, default=BACKGROUND_SIZE)
    parser.add_argument("--explain", type=int, default=EXPLAIN_SIZE)
    parser.add_argument("--nsamples", type=int, default=NSAMPLES)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model = tf.keras.models.load_model(output_dir / "full_model.keras", compile=False)
    package = joblib.load(output_dir / "nn_data_package.joblib")

    X_train = package["X_train_nn"]
    X_test = package["X_test_nn"]
    y_test = package["y_test"]
    feature_names = package["feature_names"]
    threshold = package["decision_threshold"]

    X_background, X_explain, background_idx, explain_idx = sample_rows(
        X_train, X_test, args.background, args.explain
    )
    print(f"Background {X_background.shape}, explanation {X_explain.shape}")

    shap_values = compute_shap_values(
        model, X_background, X_explain, feature_names, nsamples=args.nsamples
    )

    importance = global_importance(shap_values, feature_names)
    importance.to_csv(output_dir / "shap_global_importance.csv", index=False)
    plot_global_importance(importance, output_dir, args.top_n)
    plot_summary_scatter(shap_values, importance, feature_names, output_dir, args.top_n)
    print(importance.head(args.top_n).to_string(index=False))

    predictions = prediction_table(model, X_explain, y_test.iloc[explain_idx].values, threshold)
    predictions.to_csv(output_dir / "shap_prediction_analysis.csv", index=False)

    for case_name, row_index in select_cases(predictions).items():
        case = explain_case(
            case_name, row_index, shap_values, X_explain, feature_names, predictions
        )
        case.to_csv(output_dir / f"shap_case_{case_name}.csv", index=False)

    np.savez(
        output_dir / "shap_values.npz",
        shap_values=shap_values,
        explain_indices=explain_idx,
        background_indices=background_idx,
    )
    print(f"\nSHAP outputs written to {output_dir}")


if __name__ == "__main__":
    main()
