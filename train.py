#!/usr/bin/env python3

import multiprocessing
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from wandb_custom import CustomEpochEndWandbCallback

import sys

import numpy as np
import tensorflow as tf
from keras import mixed_precision
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import ConvLSTM2D, Dense, Flatten, Input, MaxPooling3D
from keras.models import Model
from keras.optimizers import Adam
from loguru import logger
from wandb.keras import WandbCallback, WandbMetricsLogger

import wandb
from hyperparameters import (BATCH_SIZE, DEBUG, EPOCHS, FILENAME, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE, PATIENCE, SEQUENCE_LENGTH)
from preprocess_data import get_generator_idxs, load_sequences


def setup_logging(level="DEBUG", show_module=False):
    """
    Setups better log format for loguru
    """
    logger.remove(0)    # Remove the default logger
    log_level = level
    log_fmt = u"<green>["
    log_fmt += u"{file:10.10}…:{line:<3} | " if show_module else ""
    log_fmt += u"{time:HH:mm:ss.SSS}]</green> <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, level=log_level, format=log_fmt, colorize=True, backtrace=True, diagnose=True)


def create_model(input_shape):
    """Create a ConvLSTM model."""
    inputs = Input(shape=input_shape)
    x = ConvLSTM2D(filters=4, kernel_size=(5, 5), activation="tanh", recurrent_dropout=0.2, return_sequences=True)(inputs)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Flatten()(x)
    x = Dense(4, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    out_model = Model(inputs=inputs, outputs=outputs)
    out_model.summary()
    return out_model


def main(use_wandb):
    setup_logging("DEBUG" if DEBUG else "INFO")
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"Num GPUs Available: {len(gpus)}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # Create the model
    input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    model = create_model(input_shape)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    if use_wandb:
        wandb.init(
            project="aweful-train",
            config={
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "metric": "accuracy",
                "epoch": EPOCHS,
                "batch_size": BATCH_SIZE,},
        )

    # with tf.device("CPU"):
    train_idxs = get_generator_idxs("raw", datatype="train")
    train_size = len(train_idxs)

    val_idxs = get_generator_idxs("raw", datatype="val")
    val_size = len(val_idxs)

    output_signature = (tf.TensorSpec(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.int32))

    train_dataset = tf.data.Dataset.from_generator(
        lambda: load_sequences("raw", datatype="train"),
        output_signature=output_signature,
    ).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_generator(
        lambda: load_sequences("raw", datatype="val"),
        output_signature=output_signature,
    ).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=False),
        ModelCheckpoint(FILENAME, monitor="val_accuracy", save_best_only=True, verbose=1)]

    if use_wandb:
        wandb_data = list(load_sequences("raw", datatype="wandb"))
        X_wandb = [d[0] for d in wandb_data]
        y_wandb = [d[1] for d in wandb_data]
        callbacks.append(CustomEpochEndWandbCallback(X_wandb=X_wandb, y_wandb=y_wandb))

        callbacks.append(WandbMetricsLogger())
        # callbacks.insert(0, WandbCallback(log_evaluation=True,
        #                                   generator=load_sequences("raw", datatype="wandb"),
        #                                   validation_steps=wandb_size // BATCH_SIZE,
        #                                   predictions=10,
        #                                   prediction_row_processor=lambda ndx, row: {"imgs": [wandb.Image(img) for img in row["input"]]},
        #                                   log_evaluation_frequency=1,
        #                                   verbose=2,
        #                                   save_model=False)) # yapf: disable

    # with tf.device("GPU"):
    logger.info("Training model...")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    model.fit(train_dataset,
                epochs=EPOCHS,
                callbacks=callbacks,
                validation_data=val_dataset,
                validation_steps=val_size // BATCH_SIZE,
                verbose=1)

    # Save the model weights
    model.save_weights(FILENAME)
    logger.info("Saved model weights")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(use_wandb=True)
