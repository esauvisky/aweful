#!/usr/bin/env python3

import multiprocessing
import os

from sklearn.utils import compute_class_weight

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import numpy as np
import tensorflow as tf
from keras import mixed_precision
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, ConvLSTM2D, Dropout, Flatten, Input, MaxPooling3D, MaxPooling2D, LSTM, Conv2D, Dense, Flatten, Input, LeakyReLU, Reshape, TimeDistributed, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from loguru import logger
from wandb.keras import WandbCallback, WandbMetricsLogger

import wandb
from hyperparameters import (BATCH_SIZE, DEBUG, EPOCHS, FILENAME, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE, SEQUENCE_LENGTH, DATASET_NAME)
from preprocess_data import get_batches, get_sequences
from wandb_custom import CustomEpochEndWandbCallback


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
    x = ConvLSTM2D(filters=2, kernel_size=(3, 3), activation="tanh", recurrent_dropout=0.2, return_sequences=True)(inputs)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Flatten())(x)
    x = Flatten()(x)
    x = Dense(64, activation="sigmoid")(x)
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
    tf.keras.backend.clear_session()

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

    # train_dataset = tf.data.Dataset.from_generator(
    #     lambda: get_sequences(random=True),
    #     output_types=(tf.float32, tf.uint8),
    #     output_shapes=((SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1), ())
    # ).batch(BATCH_SIZE).shuffle(10000).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = list(get_sequences(random=False))
    X_train = np.array([d[0] for d in dataset])
    y_train = np.array([d[1] for d in dataset])
    val_dataset = list(get_sequences(random=True, split_ratio=0.25))
    X_val = np.array([d[0] for d in val_dataset])
    y_val = np.array([d[1] for d in val_dataset])

    # # Calculate class weights
    # class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.flatten())
    # class_weight_dict = dict(enumerate(class_weights))

    callbacks = [
        ModelCheckpoint(FILENAME, monitor="loss", save_best_only=True, verbose=1),
        # EarlyStopping(monitor="accuracy", patience=10, restore_best_weights=False),
        ReduceLROnPlateau(monitor='accuracy', factor=0.8, patience=3, min_lr=0.00001)]

    # if use_wandb:
    #     wandb_data = list(get_sequences(random=True, split_ratio=0.001))
    #     X_wandb = [d[0] for d in wandb_data]
    #     y_wandb = [d[1] for d in wandb_data]
    #     callbacks.append(CustomEpochEndWandbCallback(X_wandb=X_wandb, y_wandb=y_wandb))
    #     callbacks.append(WandbMetricsLogger())

    with tf.device("GPU"):
        logger.info("Training model...")
        model.fit(X_train, y_train, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    # # Save the model weights
    # model.save_weights(FILENAME)
    # logger.info("Saved model weights")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(use_wandb=False)
