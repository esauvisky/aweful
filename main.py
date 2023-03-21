#!/usr/bin/env python3
import argparse
import os
import random
import sys

import keras
import numpy as np
import tensorflow as tf
import wandb
from keras import mixed_precision
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (ConvLSTM2D, Dense, Dropout, Flatten, Input, MaxPooling3D, TimeDistributed, GlobalAveragePooling2D)
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, image_utils
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from wandb.keras import WandbCallback, WandbModelCheckpoint, WandbMetricsLogger, WandbEvalCallback
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

# Define default hyperparameters
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT = 480 // 5
IMAGE_WIDTH = 640 // 5
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 5
DEBUG = False
FILENAME = f'weights-{SEQUENCE_LENGTH}_{IMAGE_HEIGHT}_{IMAGE_WIDTH}.h5'


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
    # x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = Flatten()(x)
    x = Dense(8, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    out_model = Model(inputs=inputs, outputs=outputs)
    out_model.summary()
    return out_model


class CustomBatchEndCallback(Callback):
    def __init__(self, X_train, y_train, **kwargs):
        super().__init__(**kwargs)
        self.x_train = X_train
        self.y_train = y_train
        self.reset_test_table()

    def on_train_batch_end(self, batch_ix, logs=None):
        if batch_ix % 10 == 0:
            # Get the images for the first timestep of the batch
            X_step = self.x_train[shuffled_indices[batch_ix * BATCH_SIZE]]
            y_step = "Sleep" if self.y_train[shuffled_indices[batch_ix * BATCH_SIZE]] == 1 else "Awake"

            # create a table with the images and their labels
            images = []
            for sequence_ix, image in enumerate(X_step):
                self.x_train[shuffled_indices[batch_ix * BATCH_SIZE] + sequence_ix][0]
                img = wandb.Image(image,
                                  caption=f"Image {shuffled_indices[batch_ix * BATCH_SIZE] + sequence_ix} - Label: {y_step}")
                images.append(img)

            self.test_table.add_data(batch_ix, y_step, *images)
            # create a wandb Artifact for each meaningful step
            # test_data_at = wandb.Artifact("test_samples_" + str(wandb.run.id), type="predictions")

            print(f" Last Sample:  from batch {batch_ix} - Label: {y_step})'")
            # log predictions table to wandb, giving it a name
            # test_data_at.add(test_table, "predictions")
            # wandb.run.log_artifact(test_data_at)
        super().on_train_batch_end(batch_ix, logs)

    def reset_test_table(self):
        columns = ['Index', 'Prediction']
        for s in range(SEQUENCE_LENGTH):
            columns.append(f'Sample {s + 1}')
        self.test_table = wandb.Table(columns=columns)

    def on_epoch_end(self, epoch_ix, logs=None):
        wandb.log({"data": self.test_table}, commit=True)
        self.reset_test_table()
        super().on_epoch_end(epoch_ix, logs)


def load_data(images_path, seq_length, image_height, image_width):
    X, y = [], []
    images = []
    labels = []
    X_aug = []
    y_aug = []
    datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)

    for file in tqdm(
            sorted(os.listdir(images_path), key=lambda x: int(x.split(".")[0].split("-")[0])),
            total=len(os.listdir(images_path)),
                                                                                               # sorted(os.listdir(images_path), key=lambda x: int(x.split(".")[0].split("-")[0]))[4000:4500],
                                                                                               # total=500,
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

            # Augmentation
            augmented_images = []
            seed = random.randint(0, 1000)
            for image in images:
                # for _ in range(1):
                transformed = datagen.random_transform(image, seed=seed)
                augmented_images.append(transformed)

            X_aug.append(np.array(augmented_images))
            y_aug.append(labels[-1])

            images.pop(0)
            labels.pop(0)

        if len(images) > seq_length:
            images.pop(0)
            labels.pop(0)
    X = np.array(X)
    y = np.array(y)
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    X = np.concatenate((X, X_aug), axis=0)
    y = np.concatenate((y, y_aug), axis=0)
    return X, y


def load_and_split_data():
    if os.path.exists(".data"):
        data = {
            name: pickle.load(open(f".data/{name}.npy", "rb"))
            for name in ["X", "y", "X_train", "y_train", "X_val", "y_val", "shuffled_indices"]}
    else:
        data = prepare_data()
        # os.system("rm .data -r")
        os.makedirs(".data")
        for name, array in data.items():
            pickle.dump(array, open(f".data/{name}.npy", "wb"))
    return data


def prepare_data():
    X, y = load_data("./data", SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    new_indices = np.random.permutation(len(X_train)).tolist()
    X_train, y_train = np.array([X_train[i] for i in new_indices]), np.array([y_train[i] for i in new_indices])
    return {
        "X": X, "y": y, "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "shuffled_indices": new_indices}


setup_logging("DEBUG" if DEBUG else "INFO")

logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
# Enable mixed precision training
# policy = mixed_precision.Policy("mixed_float16")
# mixed_precision.set_global_policy(policy)

# start a new wandb run to track this script
wandb.init(project="aweful",
           config={
               "optimizer": "adam",
               "loss": "binary_crossentropy",
               "metric": "accuracy",
               "epoch": EPOCHS,
               "batch_size": BATCH_SIZE,})

d = load_and_split_data()
X, y, X_train, y_train = d["X"], d["y"], d["X_train"], d["y_train"]
X_val, y_val = d["X_val"], d["y_val"]
shuffled_indices = d["shuffled_indices"]

# Create the ConvLSTM model
input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
model = create_model(input_shape)

# Compile the model
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

callbacks = [CustomBatchEndCallback(X_train, y_train), WandbCallback(save_model=True)]

with tf.device("CPU"):
    train = Dataset.from_tensor_slices((X_train, y_train)).shuffle(4*128).batch(BATCH_SIZE)
    validate = Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

model.fit(
    train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=validate,
    callbacks=callbacks,
)

# Save the model weights
model.save_weights(FILENAME)
# model.load_weights(FILENAME)

# Evaluate the model on the test set
loss, acc = model.evaluate(validate, verbose=0)
logger.info(f"Validation accuracy: {acc:.4f}, loss: {loss:.4f}")


# Make predictions on the test set
y_pred = model.predict(X_val)
y_pred_classes = np.round(y_pred)

# Evaluate the model performance
logger.info("\nConfusion matrix:\n" + str(confusion_matrix(y_val, y_pred_classes)))
logger.info("\nClassification report:\n" + str(classification_report(y_val, y_pred_classes)))


# Predict on the entire dataset to look for patterns
with tf.device("CPU"):
    X_predict = Dataset.from_tensor_slices(X).batch(BATCH_SIZE)

model_predictions = model.predict(X_predict)
model_predictions = (model_predictions > 0.5).astype(int)

# check and show results

i = 0
for i, (prediction, actual) in enumerate(zip(model_predictions, y)):
    if prediction != actual:
        logger.warning(f"{i} PRED: {'sleep\t' if prediction else 'not sleep\t'} ACTUAL: {'sleep\t' if actual else 'not sleep\t'}")
    else:
        logger.success(f"{i} {'sleep\t' if prediction else 'not sleep\t'}")
    i += 1

wandb.finish()
