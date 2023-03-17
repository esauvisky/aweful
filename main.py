#!/usr/bin/env python3
import argparse
import os
import sys

import keras
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import ConvLSTM2D, Dense, Dropout, Flatten, Input, MaxPooling3D, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, image_utils
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define default hyperparameters
SEQUENCE_LENGTH = 15
IMAGE_HEIGHT = 480 // 8
IMAGE_WIDTH = 640 // 8
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
PATIENCE = 3
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
    x = ConvLSTM2D(filters=4, kernel_size=(3, 3), activation="tanh", recurrent_dropout=0.2, return_sequences=True)(inputs)
    x = Flatten()(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


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
                keep_aspect_ratio=True,
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


def create_datagen():
    return ImageDataGenerator(rotation_range=10,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.1,
                              zoom_range=0.1,
                              horizontal_flip=True,
                              fill_mode="nearest")


def apply_augmentation(X, datagen, batch_size):
    X_augmented = []
    for i in range(len(X)):
        images = X[i]
        augmented = np.stack([datagen.random_transform(image) for image in images], axis=0)
        X_augmented.append(augmented)
    return np.array(X_augmented)


if __name__ == "__main__":
    args = parse_args()
    setup_logging("DEBUG" if args.debug else "INFO")

    # Load the data and labels
    X, y = load_data(args.data_dir, args.seq_length, args.image_height, args.image_width)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

    # Apply data augmentation
    datagen = create_datagen()
    X_train = apply_augmentation(X_train, datagen, args.batch_size)

    # Create the ConvLSTM model
    input_shape = (args.seq_length, args.image_height, args.image_width, 1)
    model = create_model(input_shape)

    # Compile the model
    optimizer = Adam(learning_rate=args.lr)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Define the callbacks
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss",
                          factor=0.1,
                          patience=args.patience,
                          verbose=1,
                          mode="auto",
                          min_delta=0.0001),
        EarlyStopping(monitor="val_loss", patience=args.patience, verbose=1, mode="auto", min_delta=0.0001),]

    if not os.path.exists(FILENAME):
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
    else:
        model.load_weights(FILENAME)

    # Evaluate the model on the test set
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Validation accuracy: {acc:.4f}, loss: {loss:.4f}")

    # Make predictions on the test set
    y_pred = model.predict(X_val)
    y_pred_classes = np.round(y_pred)

    # Evaluate the model performance
    logger.info("\nConfusion matrix:\n" + str(confusion_matrix(y_val, y_pred_classes)))
    logger.info("\nClassification report:\n" + str(classification_report(y_val, y_pred_classes)))

    model_predictions = model.predict(X_val)
    model_predictions = (model_predictions > 0.5).astype(int)

    # check and show results
    last_result = 0
    for i in range(len(model_predictions)):
        if model_predictions[i] != y_val[i]:
            logger.warning(f"{i} was predicted {'sleep' if model_predictions[i] else 'not sleep'} and is actually {'sleep' if y_val[i] else 'not sleep'}")
