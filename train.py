#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import numpy as np
import tensorflow as tf
import wandb
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import ConvLSTM2D, Dense, Flatten, Input, MaxPooling3D
from keras.models import Model
from keras.optimizers import Adam
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback

from preprocess_data import load_data
from wandb_custom import CustomBatchEndCallback

from hyperparameters import SEQUENCE_LENGTH, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, EPOCHS, LEARNING_RATE, PATIENCE, DEBUG, FILENAME


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


if __name__ == "__main__":
    setup_logging("DEBUG" if DEBUG else "INFO")
    logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    # Create the model
    input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    model = create_model(input_shape)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Start wandb
    wandb.init(
        project="aweful-train",
        config={
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metric": "accuracy",
            "epoch": EPOCHS,
            "batch_size": BATCH_SIZE,},
    )

    with tf.device("GPU"):
        X, y = load_data("./prep/")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

        callbacks = [
            CustomBatchEndCallback(X, y),
            WandbCallback(log_weights=True,
                          log_evaluation=True,
                          log_gradients=True,
                          training_data=(X_train, y_train),
                          validation_data=(X_val, y_val),
                          log_batch_frequency=10,
                          log_evaluation_frequency=10,
                          log_weights_frequency=10,
                          save_model=False),
            EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            ModelCheckpoint(FILENAME, monitor="val_accuracy", save_best_only=True, verbose=1)]

        logger.info("Training model...")
        model.fit(X_train,
                  y_train,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  callbacks=callbacks,
                  validation_data=(X_val, y_val),
                  verbose=1,
                  use_multiprocessing=True,
                  workers=32)

    # Make predictions on the test set
    y_pred = model.predict(X_val)
    y_pred_classes = np.round(y_pred)

    # Evaluate the model performance
    logger.info("\nConfusion matrix:\n" + str(confusion_matrix(y_val, y_pred_classes)))
    logger.info("\nClassification report:\n" + str(classification_report(y_val, y_pred_classes)))

    # Save the model weights
    model.save_weights(FILENAME)
    logger.info("Saved model weights")

    if os.path.exists(FILENAME):
        model.load_weights(FILENAME)
        logger.success(f"Loaded model weights from {FILENAME}")
    else:
        logger.error(f"Could not find file '{FILENAME}'")
        sys.exit(1)

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

    wandb.finish()
