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
from tensorflow.data import Dataset
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (ConvLSTM2D, Dense, Dropout, Flatten, Input, MaxPooling3D, TimeDistributed, GlobalAveragePooling2D)
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, image_utils
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from wandb.keras import WandbCallback, WandbModelCheckpoint, WandbMetricsLogger, WandbEvalCallback
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow.keras.backend as K

from tqdm import tqdm
import pickle
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

# Define default hyperparameters
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT = 480 // 10
IMAGE_WIDTH = 640 // 10
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
    x = ConvLSTM2D(filters=2, kernel_size=(5, 5), activation="tanh", recurrent_dropout=0.2, return_sequences=True)(inputs)
    # x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    # x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = Flatten()(x)
    x = Dense(4, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    out_model = Model(inputs=inputs, outputs=outputs)
    out_model.summary()
    return out_model


class CustomBatchEndCallback(Callback):
    def __init__(self, X, y, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.reset_test_table()

    def on_train_batch_end(self, batch_ix, logs=None):
        super().on_train_batch_end(batch_ix, logs)
        if batch_ix % 10 == 0:
            # Get the X value (array of images) for the first sample in the batch
            X_step = self.X[batch_ix * BATCH_SIZE]
            y_step = "Sleep" if self.y[batch_ix * BATCH_SIZE] == 1 else "Awake"

            # create a table with each image of the sequence
            images = []
            for sequence_ix, image in enumerate(X_step):
                # image is the same as self.x_train[batch_ix * BATCH_SIZE + ix][0]
                img = wandb.Image(image, caption=f"Image {batch_ix*BATCH_SIZE + sequence_ix} - Label: {y_step}")
                images.append(img)

            # Adds the row to the table with the actual index of the first image of the sequence,
            # the original label y, and the images
            self.test_table.add_data(batch_ix * BATCH_SIZE, y_step, *images)
            print(f" | Samples {batch_ix*BATCH_SIZE}-{batch_ix*BATCH_SIZE + SEQUENCE_LENGTH - 1} - Label: {y_step})'")


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
    datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)

    for file in tqdm(sorted(os.listdir(images_path), key=lambda x: int(x.split(".")[0].split("-")[0])), total=len(os.listdir(images_path))): # yapf: disable
        if len(X) % (seq_length*10) == 0:
            logger.info(f"Loaded {len(X)} samples. Restarting augmentation seed.")
            seed = random.randint(0, 1000)
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


def prepare_data():
    if os.path.exists(".data"):
        X, y = np.load(".data/data.npz")["arr_0"], np.load(".data/data.npz")["arr_1"]
    else:
        X, y = load_data("./data", SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)
        os.makedirs(".data", exist_ok=True)
        np.savez(".data/data.npz", X, y)
    # _indices = shuffle(range(len(X)), random_state=41)
    # X_train, y_train = shuffle(X, y, random_state=41)
    # X_val, y_val = shuffle(X, y, random_state=41)
    indices = np.random.permutation(len(X)).tolist()
    X_train, y_train = X[indices], y[indices]
    X_val, y_val = X[indices], y[indices]
    X_train = X_train[:int(len(X) * 0.95)]
    y_train = y_train[:int(len(y) * 0.95)]
    X_val = X_val[int(len(X) * 0.95):]
    y_val = y_val[int(len(y) * 0.95):]
    return {"X": X, "y": y, "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}

def data_generator(X, y, batch_size):
    num_samples = len(X)
    while True:
        # Shuffle the data at the start of each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        # Generate batches of data
        for start in range(0, num_samples, batch_size):
            if start + batch_size > num_samples:
                break
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]
            yield X_batch, y_batch

setup_logging("DEBUG" if DEBUG else "INFO")
# logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Wrap the input pipeline with a Cast layer
if policy.name == 'mixed_float16':
    cast_dtype = 'float16'
else:
    cast_dtype = 'float32'

# start a new wandb run to track this script
wandb.init(project="aweful",
           config={
               "optimizer": "adam",
               "loss": "binary_crossentropy",
               "metric": "accuracy",
               "epoch": EPOCHS,
               "batch_size": BATCH_SIZE,})

d = prepare_data()
X, y, X_train, y_train = d["X"], d["y"], d["X_train"], d["y_train"]
X_val, y_val = d["X_val"], d["y_val"]

# Create the ConvLSTM model
input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
model = create_model(input_shape)

# Compile the model
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# K.set_learning_phase(False)

train_dataset = tf.data.Dataset.from_generator(generator=data_generator,
                                                args=(X_train, y_train, BATCH_SIZE),
                                                output_types=(tf.float16, tf.int16),
                                                output_shapes=((BATCH_SIZE, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH,
                                                                1), (BATCH_SIZE,))).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(generator=data_generator,
                                                args=(X_val, y_val, BATCH_SIZE),
                                                output_types=(tf.float16, tf.int16),
                                                output_shapes=((BATCH_SIZE, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH,
                                                                1), (BATCH_SIZE,))).prefetch(tf.data.experimental.AUTOTUNE)


callbacks = [
    CustomBatchEndCallback(X, y),
    WandbCallback(log_weights=True, log_evaluation=True, log_batch_frequency=10, log_evaluation_frequency=10, log_weights_frequency=10, save_model=False),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(FILENAME, monitor="val_loss", save_best_only=True, verbose=1)]

# if os.path.exists(FILENAME):
#     model.load_weights(FILENAME)

# Compute the number of steps per epoch
steps_per_epoch = len(X_train) // BATCH_SIZE
validation_steps = len(X_val) // BATCH_SIZE

model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
)

# Save the model weights
model.save_weights(FILENAME)

# # Evaluate the model on the test set
# loss, acc = model.evaluate(val_dataset, verbose=2)
# logger.info(f"Validation accuracy: {acc:.4f}, loss: {loss:.4f}")

# # Make predictions on the test set
# y_pred = model.predict(X_val)
# y_pred_classes = np.round(y_pred)

# # Evaluate the model performance
# logger.info("\nConfusion matrix:\n" + str(confusion_matrix(y_val, y_pred_classes)))
# logger.info("\nClassification report:\n" + str(classification_report(y_val, y_pred_classes)))

# # Predict on the entire dataset to look for patterns
# # with tf.device("CPU"):
# X_predict = Dataset.from_tensor_slices(X).batch(BATCH_SIZE)

model_predictions = model.predict(X, batch_size=BATCH_SIZE, verbose=1)
model_predictions = (model_predictions > 0.5).astype(int)

# check and show results

i = 0
for i, (prediction, actual) in enumerate(zip(model_predictions, y)):
    if prediction != actual:
        logger.warning(f"{i} PRED: {'sleep' if prediction else 'not sleep'}\t ACTUAL: {'sleep' if actual else 'not sleep'}")
    else:
        logger.success(f"{i} {'sleep' if prediction else 'not sleep'}")
    i += 1

wandb.finish()
