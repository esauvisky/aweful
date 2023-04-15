#!/usr/bin/env python3

import os

from tqdm.auto import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import numpy as np
import tensorflow as tf
from keras import mixed_precision
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import ConvLSTM2D, Dense, Flatten, Input, MaxPooling3D
from keras.models import Model
from keras.optimizers import Adam
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback

import wandb
from hyperparameters import (BATCH_SIZE, DEBUG, EPOCHS, FILENAME, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE, PATIENCE, SEQUENCE_LENGTH)
from preprocess_data import load_individual_data
from wandb_custom import CustomBatchEndCallback


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
    x = ConvLSTM2D(filters=2, kernel_size=(5, 5), activation="tanh", recurrent_dropout=0.2, return_sequences=True)(inputs)
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

    with tf.device("CPU"):
        # Call the new function with the appropriate key
        X, y = load_individual_data("raw")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(BATCH_SIZE)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.shuffle(buffer_size=len(X_val)).batch(BATCH_SIZE)
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=False),
        ModelCheckpoint(FILENAME, monitor="val_accuracy", save_best_only=True, verbose=1)]

    if use_wandb:
        callbacks.insert(0, CustomBatchEndCallback(X, y))
        callbacks.insert(0, WandbCallback(log_weights=True,
                                          log_evaluation=True,
                                          log_batch_frequency=len(X_train) / BATCH_SIZE / 10,
                                          log_evaluation_frequency=len(X_train) / BATCH_SIZE / 10,
                                          log_weights_frequency=len(X_train) / BATCH_SIZE / 10,
                                          save_model=False)) # yapf: disable

    with tf.device("GPU"):
        logger.info("Training model...")
        model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks, validation_data=val_dataset, verbose=1)

    # Make predictions on the test set
    y_pred = model.predict(X_val)
    y_pred_classes = np.round(y_pred)

    # Evaluate the model performance
    logger.info("\nConfusion matrix:\n" + str(confusion_matrix(y_val, y_pred_classes)))
    logger.info("\nClassification report:\n" + str(classification_report(y_val, y_pred_classes)))

    # Save the model weights
    model.save_weights(FILENAME)
    logger.info("Saved model weights")

    for index in range(0, 1000):
        # picks a random X value
        # index = random.randint(0, len(X) - 1)
        X_predict = X[index:index + 1]
        y_predict = y[index:index + 1]
        y_out = model.predict(X_predict, verbose=0)
        y_out = np.round(y_out).flatten().astype(int)
        for n, cat in enumerate(y_out):
            prediction = "🆙" if cat == 0 else "💤"
            if y_predict[n] != cat: color = "\033[91m"
            else: color = "\033[92m" if cat == 0 else "\033[93m"
            print(color + f"{index:05d}:" + str(prediction) + "\033[0m", end="\t ")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(use_wandb=False)
