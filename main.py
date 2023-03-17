#!/usr/bin/env python3
import argparse
import os
import sys

import random
import keras
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import ConvLSTM2D, Dense, Dropout, Flatten, Input, MaxPooling3D, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, image_utils
from loguru import logger
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from keras import mixed_precision

# Define default hyperparameters
SEQUENCE_LENGTH = 15
IMAGE_HEIGHT = 480 // 10
IMAGE_WIDTH = 640 // 10
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 5
FILENAME = f'weights-{SEQUENCE_LENGTH}_{IMAGE_HEIGHT}_{IMAGE_WIDTH}.h5'


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a ConvLSTM model on video frames.")
    parser.add_argument("--data-dir", type=str, default="./data/", help="Directory containing the data.")
    parser.add_argument("--seq-length", type=int, default=SEQUENCE_LENGTH, help="Sequence length.")
    parser.add_argument("--image-height", type=int, default=IMAGE_HEIGHT, help="Image height.")
    parser.add_argument("--image-width", type=int, default=IMAGE_WIDTH, help="Image width.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--patience",
                        type=int,
                        default=PATIENCE,
                        help="Number of epochs with no improvement to wait before reducing the learning rate.")
    parser.add_argument("--filename",
                        type=str,
                        default=FILENAME,
                        help="Name of the file to save the trained model weights.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    return parser.parse_args()


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


def load_data(images_path, seq_length, image_height, image_width):
    """Load the data and labels."""
    X, y = [], []
    images = []
    labels = []
    for file in tqdm(
            sorted(os.listdir(images_path), key=lambda x: int(x.split(".")[0].split("-")[0])),
            total=len(os.listdir(images_path)),
    ):
        if file.endswith(".jpg"):
            logger.debug(f"Loading {file}")
            image = image_utils.load_img(
                os.path.join(images_path, file),
                target_size=(image_height, image_width),
                color_mode="grayscale",
            )
            image = image_utils.img_to_array(image) / 255.0
            images.append(image)
            if "sleep" in file:
                labels.append(1)
            else:
                labels.append(0)
        if len(images) == seq_length:
            X.append(np.array(images))
            y.append(labels[-1])
            images.pop(0)
            labels.pop(0)
        if len(images) > seq_length:
            images.pop(0)
            labels.pop(0)
    X = np.array(X)
    y = np.array(y)
    return X, y


def apply_augmentation(X, n_augmentations):
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)
    X_augmented = []
    for i in tqdm(range(len(X))):
        images = X[i]
        seed = random.randint(0, 1000)
        augmented_sequences = [
            np.stack([datagen.random_transform(image, seed=seed) for image in images], axis=0) for _ in range(n_augmentations)]
        X_augmented.extend(augmented_sequences)
    return np.array(X_augmented)


if __name__ == "__main__":
    args = parse_args()
    setup_logging("DEBUG" if args.debug else "INFO")

    # Enable mixed precision training
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)

    if os.path.exists(".data"):
        X, y = np.load(".data/X.npy"), np.load(".data/y.npy")
        X_train, y_train = np.load(".data/X_train.npy"), np.load(".data/y_train.npy")
        X_val, y_val = np.load(".data/X_val.npy"), np.load(".data/y_val.npy")
    else:
        X, y = load_data(args.data_dir, args.seq_length, args.image_height, args.image_width)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

        X_train_aug = apply_augmentation(X_train, n_augmentations=1)
        y_train_aug = np.tile(y_train, 1) # Repeat the labels for the augmented sequences

        X_train = np.concatenate((X_train, X_train_aug), axis=0)
        y_train = np.concatenate((y_train, y_train_aug), axis=0)
        os.makedirs(".data")
        np.save(".data/X.npy", X)
        np.save(".data/y.npy", y)
        np.save(".data/X_train.npy", X_train)
        np.save(".data/y_train.npy", y_train)
        np.save(".data/X_val.npy", X_val)
        np.save(".data/y_val.npy", y_val)

    # Create the ConvLSTM model
    input_shape = (args.seq_length, args.image_height, args.image_width, 1)
    model = create_model(input_shape)

    # Compile the model
    optimizer = Adam(learning_rate=args.lr)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Define the callbacks
    callbacks = [
                   # ReduceLROnPlateau(monitor="val_loss",
                   #                   factor=0.01,
                   #                   patience=args.patience,
                   #                   verbose=1,
                   #                   mode="auto",
                   #                   min_delta=0.00005),
                   # EarlyStopping(monitor="val_loss", patience=args.patience, verbose=1, mode="min", min_delta=0.00001)
    ]

    if os.path.exists(FILENAME):
        model.load_weights(FILENAME)

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    # Save the model weights
    model.save_weights(args.filename)

    # Evaluate the model on the test set
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Validation accuracy: {acc:.4f}, loss: {loss:.4f}")

    # Make predictions on the test set
    y_pred = model.predict(X_val)
    y_pred_classes = np.round(y_pred)

    # Evaluate the model performance
    logger.info("\nConfusion matrix:\n" + str(confusion_matrix(y_val, y_pred_classes)))
    logger.info("\nClassification report:\n" + str(classification_report(y_val, y_pred_classes)))

    # shuffle X and y
    # permutation = np.random.permutation(X.shape[0])
    # X = X[permutation][:5000]
    # y = y[permutation][:5000]
    model_predictions = model.predict(X)
    model_predictions = (model_predictions > 0.5).astype(int)

    # check and show results
    last_result = 0
    for i in range(len(model_predictions)):
        if model_predictions[i] != y[i]:
            logger.warning(f"{i} was predicted {'sleep' if model_predictions[i] else 'not sleep'} and is actually {'sleep' if y[i] else 'not sleep'}")
        else:
            logger.success(f"{i} was predicted {'sleep' if model_predictions[i] else 'not sleep'} and is actually {'sleep' if y[i] else 'not sleep'}")
        last_result = i


