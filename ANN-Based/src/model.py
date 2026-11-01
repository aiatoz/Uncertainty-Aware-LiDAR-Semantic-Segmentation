"""
ANN architecture + training loop.
"""
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger

logger = get_logger(__name__)


def build_model(input_dim: int, num_classes: int, config: dict) -> keras.Model:
    arch = config["training"]["architecture"]
    dropout_rate = arch.get("dropout", 0.0)

    layers = [keras.layers.Input(shape=(input_dim,))]
    for units in arch["hidden_layers"]:
        layers.append(keras.layers.Dense(units, activation=arch.get("activation", "relu")))
        if dropout_rate > 0:
            layers.append(keras.layers.Dropout(dropout_rate))
    layers.append(keras.layers.Dense(num_classes, activation="softmax"))

    model = keras.Sequential(layers)
    return model


def compile_model(model: keras.Model, config: dict, loss) -> keras.Model:
    model.compile(
        optimizer=config["training"].get("optimizer", "adam"),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def make_tf_dataset(X, y, batch_size: int, shuffle: bool = True, shuffle_buffer: int = 200_000, sample_weight=None, seed: int = 42):
    import numpy as np
    import tensorflow as tf

    n = len(X)
    feature_dim = X.shape[1]
    rng = np.random.default_rng(seed)

    def gen():
        order = rng.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            batch_idx = order[start:start + batch_size]
            if sample_weight is None:
                yield X[batch_idx], y[batch_idx]
            else:
                yield X[batch_idx], y[batch_idx], sample_weight[batch_idx]

    if sample_weight is None:
        output_signature = (
            tf.TensorSpec(shape=(None, feature_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=(None, feature_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    steps_per_epoch = -(-n // batch_size)  # ceil division
    ds = ds.apply(tf.data.experimental.assert_cardinality(steps_per_epoch))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds



def train_model(model: keras.Model, X_train, y_onehot_train, config: dict, sample_weight=None, checkpoint_path=None, X_val=None, y_val=None):
    import tensorflow as tf

    train_cfg = config["training"]
    es_cfg    = config["training"].get("early_stopping", {})

    is_dataset = isinstance(X_train, tf.data.Dataset)
    has_val = X_val is not None and (y_val is not None or isinstance(X_val, tf.data.Dataset))
    monitor  = "val_loss" if has_val else "loss"

    callbacks = []
    if es_cfg.get("enabled", False):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=es_cfg.get("patience", 5),
                restore_best_weights=True,
                verbose=1,
            )
        )
    if checkpoint_path is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                str(checkpoint_path), save_best_only=True, monitor=monitor, verbose=1,
            )
        )

    if not has_val:
        logger.info(
            "No validation data provided — monitoring training loss. "
            "This avoids the memory/stall issue caused by validation_split "
            "on large datasets (see ISSUES.md T-1). Pass X_val/y_val if "
            "you want val_loss monitoring from a dedicated validation file."
        )

    if is_dataset:
        logger.info(
            "X_train is a tf.data.Dataset — using it directly (already "
            "batched/shuffled/prefetched by make_tf_dataset()); "
            "y_onehot_train, sample_weight and config batch_size are ignored."
        )
        fit_kwargs = dict(
            epochs=train_cfg.get("num_epochs", 30),
            callbacks=callbacks,
            verbose=1,
        )
        if has_val:
            fit_kwargs["validation_data"] = X_val
        history = model.fit(X_train, **fit_kwargs)
        return model, history

    fit_kwargs = dict(
        sample_weight=sample_weight,
        epochs=train_cfg.get("num_epochs", 30),
        batch_size=train_cfg.get("batch_size", 512),
        callbacks=callbacks,
        verbose=1,
    )
    if has_val:
        fit_kwargs["validation_data"] = (X_val, y_val)

    history = model.fit(X_train, y_onehot_train, **fit_kwargs)
    return model, history