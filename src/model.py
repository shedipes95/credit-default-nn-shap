"""The Keras architecture and the three topologies compared in the sweep.

Every hidden block is Dense(relu) -> BatchNormalization -> Dropout(0.3). The
output is a single sigmoid unit. The model is compiled with Adam, binary
cross-entropy, and AUC alongside accuracy, because accuracy on an 8%-positive
problem is close to meaningless on its own.
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

TOPOLOGIES: dict[str, list[int]] = {
    "Topology_1_Small": [64, 32],
    "Topology_2_Medium": [128, 64, 32],
    "Topology_3_Large": [256, 128, 64, 32],
}

BATCH_SIZES: list[int] = [256, 512, 1024]

DROPOUT_RATE = 0.3
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 3


def build_nn_model(input_dim: int, layers: list[int]) -> Sequential:
    """Build and compile the feed-forward network for one topology.

    ``layers`` is the hidden-unit count per layer, e.g. ``[64, 32]``.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for units in layers:
        model.add(Dense(units, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def early_stopping(patience: int = EARLY_STOPPING_PATIENCE) -> EarlyStopping:
    """Stop when validation AUC stops improving, and keep the best weights.

    AUC rather than loss, for the same reason accuracy is not the headline
    metric here: the class balance makes loss a poor stopping signal.
    """
    return EarlyStopping(
        monitor="val_auc",
        patience=patience,
        mode="max",
        restore_best_weights=True,
    )


def set_seed(seed: int = 42) -> None:
    """Seed TensorFlow so a rerun lands in roughly the same place."""
    tf.keras.utils.set_random_seed(seed)
