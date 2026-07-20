from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

from matched_view_eval.errors import PipelineInvariantError
from matched_view_eval.training_config import TrainingConfig


def build_random_forest(config: TrainingConfig) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=config.random_forest.n_estimators,
        max_depth=config.random_forest.max_depth,
        n_jobs=-1,
        class_weight=config.random_forest.class_weight,
        random_state=config.seed,
    )


def build_xgboost(config: TrainingConfig, *, representation: str) -> Any:
    from xgboost import XGBClassifier

    learning_rates = {
        "matched_flow_stats": config.xgboost.matched_flow_stats_learning_rate,
        "flattened_splt": config.xgboost.flattened_splt_learning_rate,
    }
    if representation not in learning_rates:
        raise PipelineInvariantError(f"Unsupported XGBoost representation: {representation}")
    return XGBClassifier(
        n_estimators=config.xgboost.n_estimators,
        max_depth=config.xgboost.max_depth,
        learning_rate=learning_rates[representation],
        subsample=config.xgboost.subsample,
        colsample_bytree=config.xgboost.colsample_bytree,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=config.seed,
        verbosity=0,
    )


def configure_tensorflow(seed: int, *, device: str) -> Any:
    """Import TensorFlow only after deterministic runtime controls are set."""
    if device not in {"auto", "cpu", "gpu"}:
        raise PipelineInvariantError(f"Unsupported TensorFlow device: {device}")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError as error:
        raise PipelineInvariantError(
            "CNN1D execution requires the pinned TensorFlow dependency"
        ) from error

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not gpus:
        raise PipelineInvariantError("GPU execution was requested but TensorFlow found no GPU")
    return tf


def make_sparse_focal_loss(tf: Any, *, gamma: float, alpha: float) -> Any:
    """Return one focal-loss value per sample so Keras can apply class weights."""

    def sparse_focal_loss(y_true: Any, y_pred: Any) -> Any:
        labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        probabilities = tf.reduce_sum(
            y_pred * tf.one_hot(labels, tf.shape(y_pred)[-1]), axis=-1
        )
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(
            labels, y_pred, from_logits=False
        )
        return alpha * tf.pow(1.0 - probabilities, gamma) * cross_entropy

    sparse_focal_loss.__name__ = "sparse_focal_loss"
    return sparse_focal_loss


def build_cnn1d(config: TrainingConfig, tf: Any) -> Any:
    keras = tf.keras
    layers = keras.layers
    cnn = config.cnn1d
    inputs = keras.Input(
        shape=(config.dataset.primary_prefix_length, 3), name="splt_input"
    )
    branches = [
        layers.Conv1D(64, kernel, padding="same", activation="relu", name=f"conv_k{kernel}")(
            inputs
        )
        for kernel in (3, 7, 11)
    ]
    x = layers.Concatenate(name="multi_scale_concat")(branches)
    x = layers.BatchNormalization(name="bn1")(x)
    skip = layers.Conv1D(192, 1, padding="same", name="res_skip")(x)
    residual = layers.Conv1D(
        192, 3, padding="same", activation="relu", name="res_conv3"
    )(x)
    residual = layers.BatchNormalization(name="bn_res")(residual)
    x = layers.Add(name="res_add")([residual, skip])
    x = layers.Activation("relu", name="res_relu")(x)
    attention = layers.MultiHeadAttention(
        num_heads=4, key_dim=48, dropout=0.1, name="mha"
    )(x, x)
    x = layers.Add(name="attn_add")([x, attention])
    x = layers.LayerNormalization(name="ln1")(x)
    x = layers.Conv1D(256, 3, padding="same", activation="relu", name="conv_final")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(cnn.dropout, name="drop1")(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.Dropout(cnn.dropout / 2.0, name="drop2")(x)
    outputs = layers.Dense(
        len(config.dataset.expected_classes), activation="softmax", name="output"
    )(x)
    model = keras.Model(inputs, outputs, name="MultiScale_CNN1D_Attention")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cnn.learning_rate),
        loss=make_sparse_focal_loss(tf, gamma=cnn.focal_gamma, alpha=cnn.focal_alpha),
        metrics=["accuracy"],
    )
    trainable = int(sum(np.prod(variable.shape) for variable in model.trainable_weights))
    if model.count_params() != cnn.expected_total_parameters:
        raise PipelineInvariantError(
            f"CNN1D has {model.count_params()} total parameters; "
            f"expected {cnn.expected_total_parameters}"
        )
    if trainable != cnn.expected_trainable_parameters:
        raise PipelineInvariantError(
            f"CNN1D has {trainable} trainable parameters; "
            f"expected {cnn.expected_trainable_parameters}"
        )
    return model
