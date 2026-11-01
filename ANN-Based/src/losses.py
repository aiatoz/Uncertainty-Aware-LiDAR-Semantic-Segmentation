"""
Loss functions
"""
import numpy as np
import tensorflow as tf


def uncertainty_to_sample_weights(epistemic_uncertainty: np.ndarray,
                                   u_min: float = None,
                                   u_max: float = None) -> np.ndarray:
    """
    Implements equation:
        u_tilde_i = (u_i - min_j u_j) / (max_j u_j - min_j u_j)
        w_i = 1 - u_tilde_i
    Returns w_i, to be passed as `sample_weight` to model.fit()

    u_min/u_max should be computed on the TRAIN split and reused as-is for any other split, to avoid leaking that split's own uncertainty range into its own normalization.
    """
    u_min = epistemic_uncertainty.min() if u_min is None else u_min
    u_max = epistemic_uncertainty.max() if u_max is None else u_max
    u_tilde = (epistemic_uncertainty - u_min) / (u_max - u_min + 1e-12)
    w = 1.0 - u_tilde
    return w


def get_categorical_crossentropy_loss() -> tf.keras.losses.Loss:
    return tf.keras.losses.CategoricalCrossentropy()


def get_sparse_categorical_crossentropy_loss() -> tf.keras.losses.Loss:
    return tf.keras.losses.SparseCategoricalCrossentropy()


def weighted_categorical_crossentropy_correct(y_true, y_pred, sample_weight):

    cce = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    per_sample_loss = cce(y_true, y_pred)
    weighted = per_sample_loss * sample_weight
    return tf.reduce_mean(weighted)