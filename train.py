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
from keras.layers import MaxPooling3D, LSTM, LeakyReLU, Reshape, BatchNormalization
from keras.optimizers import Adam, SGD
from loguru import logger
from wandb.keras import WandbCallback, WandbMetricsLogger

import wandb
from hyperparameters import BATCH_SIZE, DEBUG, EPOCHS, FILENAME, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE, SEQUENCE_LENGTH, DATASET_NAME, create_model
from preprocess_data import process_data
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


def main(use_wandb):
    setup_logging("DEBUG" if DEBUG else "INFO")
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"Num GPUs Available: {len(gpus)}")
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)
    # tf.keras.backend.clear_session()

    # Create the model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model = create_model()
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # if use_wandb:
    #     wandb.init(
    #         project="aweful-train",
    #         config={
    #             "optimizer": "adam",
    #             "loss": "binary_crossentropy",
    #             "metric": "accuracy",
    #             "epoch": EPOCHS,
    #             "batch_size": BATCH_SIZE,},
    #     )

    # train_dataset = tf.data.Dataset.from_generator(lambda: process_data("./data/quick"),
    #                                                output_types=(tf.float16, tf.float16),
    #                                                output_shapes=((SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1), ())).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    data_iter = process_data("./data/raw")
    # X = tf.convert_to_tensor(data_iter[0], dtype=tf.float16)
    # y = tf.convert_to_tensor(data_iter[1], dtype=tf.float16)
    X = data_iter[0]
    y = data_iter[1]

    callbacks = [
        ModelCheckpoint(FILENAME, monitor="loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=5, min_lr=0.00001, min_delta=0.001)]

    logger.info("Training model...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1, shuffle=True)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(use_wandb=False)
