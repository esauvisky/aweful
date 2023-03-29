#!/usr/bin/env python3

import datetime
import os
import random
import shlex
import subprocess
import sys
from collections import deque
import cv2

from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import numpy as np
import cupy as cp
import tensorflow as tf
from keras import mixed_precision
from keras.callbacks import (Callback, EarlyStopping, ModelCheckpoint)
from keras.layers import (ConvLSTM2D, Dense, Flatten, Input, MaxPooling3D)
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, image_utils
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from tqdm import tqdm
from wandb.keras import WandbCallback

import wandb

# Define default hyperparameters
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT = 480 // 2
IMAGE_WIDTH = 640 // 2
BATCH_SIZE = 10
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
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    # x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = Flatten()(x)
    x = Dense(4, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    out_model = Model(inputs=inputs, outputs=outputs)
    out_model.summary()
    return out_model


def create_wandb_table():
    columns = ['Index', 'Date', 'Prediction']
    for s in range(SEQUENCE_LENGTH):
        columns.append(f'Sample {s + 1}')
    return wandb.Table(columns=columns, allow_mixed_types=True)


class CustomBatchEndCallback(Callback):
    def __init__(self, X, y, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.test_table = create_wandb_table()

    def on_train_batch_end(self, batch_ix, logs=None):
        super().on_train_batch_end(batch_ix, logs)
        if batch_ix % 10 == 0:
            # Get the X value (array of images) for the first sample in the batch
            X_step = self.X[batch_ix * BATCH_SIZE]
            y_step = "Sleep" if self.y[batch_ix * BATCH_SIZE] == 1 else "Awake"

            # create a table with each image of the sequence
            images = []
            for sequence_ix, _img in enumerate(X_step):
                # image is the same as self.x_train[batch_ix * BATCH_SIZE + ix][0]
                img = wandb.Image(_img, caption=f"Image {batch_ix*BATCH_SIZE + sequence_ix} - Label: {y_step}")
                images.append(img)

            # Adds the row to the table with the actual index of the first image of the sequence,
            # the original label y, and the images
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.test_table.add_data(batch_ix * BATCH_SIZE, date, y_step, *images)
            print(f" | Samples {batch_ix*BATCH_SIZE}-{batch_ix*BATCH_SIZE + SEQUENCE_LENGTH - 1} - Label: {y_step})'")

    def on_epoch_end(self, epoch, logs=None):
        wandb.log({"data": self.test_table}, commit=True)
        self.test_table = create_wandb_table()
        super().on_epoch_end(epoch, logs)


def get_random_crop(image, seed=None):
    if seed:
        np.random.seed(seed)

    height, width = image.shape[0], image.shape[1]
    aspect_ratio = float(width) / float(height)

    if width > height:
        new_width = np.random.randint(int(width * 0.8), width)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = np.random.randint(int(height * 0.8), height)
        new_width = int(new_height * aspect_ratio)

    x = np.random.randint(0, width - new_width)
    y = np.random.randint(0, height - new_height)

    crop = image[y:y + new_height, x:x + new_width]
    resized_crop = cv2.resize(crop, (width, height))
    resized_crop = np.expand_dims(resized_crop, axis=-1) # Add an extra channel dimension
    return resized_crop


def get_image(image_path, image_height, image_width):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image[0:image.shape[0], 0:image.shape[1] - 21]
    image = cv2.resize(image, (image_width, image_height))
    image = np.expand_dims(image, axis=-1) # Add an extra channel dimension
    return image


def process_image_sequence(image_files, images_path, image_height, image_width):
    images = []
    labels = []
    augmented_images = []
    datagen = ImageDataGenerator(rotation_range=10)
    seed = random.randint(0, 1000)

    for file in image_files:
        image = get_image(os.path.join(images_path, file), image_height, image_width)
        label = 1 if "sleep" in file else 0
        images.append(image)
        labels.append(label)

        # Augmentation
        transformed = datagen.random_transform(image, seed=seed)
        cropped = get_random_crop(transformed, seed=seed)
        augmented_images.append(cropped)

    return images, labels[-1], augmented_images


def load_data(images_path, seq_length, image_height, image_width):
    X, y = [], []
    X_aug, y_aug = [], []

    # Get the list of image files
    image_files = sorted((file for file in os.listdir(images_path) if file.endswith(".jpg")),
                         key=lambda x: int(x.split(".")[0].split("-")[0]))

    # Use a ThreadPoolExecutor to process image sequences in parallel
    with ThreadPoolExecutor(max_workers=16) as executor:
        # Create the tqdm progress bar
        progress_bar = tqdm(total=len(image_files) - seq_length,
                            smoothing=0.1,
                            desc="Processing images...",)

        # Submit the tasks to the executor
        futures = [
            executor.submit(process_image_sequence, image_files[i:i + seq_length], images_path, image_height, image_width)
            for i in range(0,
                           len(image_files) - seq_length)]

        # Use as_completed() to process the results as they become available, and update the progress bar
        for future in as_completed(futures):
            images, label, augmented_images = future.result()
            X.append(images)
            X_aug.append(augmented_images)
            y.append(label)
            y_aug.append(label)
            progress_bar.update(1)

        progress_bar.close()

    logger.info("Data loaded. Preparing arrays...")
    X = np.array(X)
    y = np.array(y)
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    X = np.concatenate((X, X_aug), axis=0)
    y = np.concatenate((y, y_aug), axis=0)
    return X, y


def prepare_data():
    if os.path.exists(".data"):
        logger.info("Loading data from disk...")
        X, y = np.load(".data/data.npz")["arr_0"], np.load(".data/data.npz")["arr_1"]
    else:
        X, y = load_data("./data", SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)
        os.makedirs(".data", exist_ok=True)
        logger.info("Saving data to disk...")
        np.savez(".data/data.npz", X, y)
    indices = np.random.permutation(len(X)).tolist()
    X_train, y_train = X[indices], y[indices]
    X_val, y_val = X[indices], y[indices]
    X_train = X_train[:int(len(X) * 0.8)]
    y_train = y_train[:int(len(y) * 0.8)]
    X_val = X_val[int(len(X) * 0.8):]
    y_val = y_val[int(len(y) * 0.8):]
    X = np.array(X, dtype=np.float16)
    y = np.array(y, dtype=np.int8)
    return {"X": X, "y": y, "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}


def data_generator(X, y, batch_size):
    num_samples = len(X)
    while True:
        # Generate batches of data
        for start in range(0, num_samples, batch_size):
            if start + batch_size > num_samples:
                break
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]
            np.divide(X_batch, 255.0, out=X_batch)
            yield X_batch, y_batch


def set_mixed_precision():
    # Enable mixed precision training
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    # # Wrap the input pipeline with a Cast layer
    # if policy.name == 'mixed_float16':
    #     cast_dtype = 'float16'
    # else:
    #     cast_dtype = 'float32'
    # return cast_dtype


if len(sys.argv) >= 2 and sys.argv[1] == "train":
    sys.argv.pop(1)
    routine = "train"
else:
    routine = "clock"

setup_logging("DEBUG" if DEBUG else "INFO")
logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# start a new wandb run to track this script
wandb.init(project=f"aweful-{routine}",
           config={
               "optimizer": "adam",
               "loss": "binary_crossentropy",
               "metric": "accuracy",
               "epoch": EPOCHS,
               "batch_size": BATCH_SIZE,})

set_mixed_precision()
d = prepare_data()
X, y, X_train, y_train = d["X"], d["y"], d["X_train"], d["y_train"]
X_val, y_val = d["X_val"], d["y_val"]

# Create the ConvLSTM model
input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
model = create_model(input_shape)

# Compile the model
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

if os.path.exists(FILENAME):
    model.load_weights(FILENAME)
    logger.success(f"Loaded model weights from {FILENAME}")

if routine == "train":
    with tf.device("CPU"):
        train_dataset = tf.data.Dataset.from_generator(generator=data_generator,
                                                       args=(X_train, y_train, BATCH_SIZE),
                                                       output_types=(tf.float16, tf.int8),
                                                       output_shapes=(
                                                           (BATCH_SIZE, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1),
                                                           (BATCH_SIZE,),
                                                       )).prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(generator=data_generator,
                                                     args=(X_val, y_val, BATCH_SIZE),
                                                     output_types=(tf.float16, tf.int8),
                                                     output_shapes=(
                                                         (BATCH_SIZE, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1),
                                                         (BATCH_SIZE,),
                                                     )).prefetch(tf.data.experimental.AUTOTUNE)

    # Compute the number of steps per epoch
    steps_per_epoch = len(X_train) // BATCH_SIZE
    validation_steps = len(X_val) // BATCH_SIZE

    callbacks = [
        CustomBatchEndCallback(X, y),
        WandbCallback(log_weights=True,
                      log_evaluation=True,
                      log_batch_frequency=10,
                      log_evaluation_frequency=10,
                      log_weights_frequency=10,
                      validation_data=val_dataset,
                      validation_steps=validation_steps,
                      save_model=False),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(FILENAME, monitor="val_loss", save_best_only=True, verbose=1)]

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
    )
    logger.info("Finished training")

    # Save the model weights
    model.save_weights(FILENAME)

    # # Make predictions on the test set
    y_pred = model.predict(X_val)
    y_pred_classes = np.round(y_pred)

    # # Evaluate the model performance
    logger.info("\nConfusion matrix:\n" + str(confusion_matrix(y_val, y_pred_classes)))
    logger.info("\nClassification report:\n" + str(classification_report(y_val, y_pred_classes)))

    # # Predict on the entire dataset to look for patterns
    # with tf.device("CPU"):
    #     X_predict = tf.data.Dataset.from_generator(generator=data_generator,
    #                                                args=(X, y, BATCH_SIZE),
    #                                                output_types=(tf.float16, tf.int8),
    #                                                output_shapes=(
    #                                                    (BATCH_SIZE, SEQUENCE_LENGTH, IMAGE, IMAGE_WIDTH, 1),
    #                                                    (BATCH_SIZE,),
    #                                                )).prefetch(tf.data.experimental.AUTOTUNE)

# for i in range(0, len(X), BATCH_SIZE):
#     X_predict = X[i:i + BATCH_SIZE]
#     y_predict = y[i:i + BATCH_SIZE]
#     y_out = model.predict(X_predict, verbose=0)
#     y_out = np.round(y_out).flatten().astype(int)
#     for n, cat in enumerate(y_out):
#         prediction = "🆙" if cat == 0 else "💤"
#         if y_predict[n] != cat: color = "\033[91m"
#         else: color = "\033[92m" if cat == 0 else "\033[93m"
#         print(color + f"{i:05d}:" + str(prediction) + "\033[0m", end=" | ")

if routine == "clock":
    sleep_counter = deque(maxlen=6 * 60)
    images = deque(maxlen=SEQUENCE_LENGTH)
    # images_batch = deque(maxlen=BATCH_SIZE)
    path = "/tmp/image.jpg"

    main_table = create_wandb_table()
    index = 0
    while True:
        index += 1
        # Take a photo
        subprocess.run(shlex.split(f"fswebcam {path} -d /dev/video0 -S2 -F1"),
                       check=False,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

        # Load and add the preprocessed image to the sequence
        image = image_utils.load_img(path, color_mode="grayscale")
        image = image_utils.img_to_array(image) / 255.0
        image = image[:-21, :]
        resized = image_utils.array_to_img(image).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        resized = image_utils.img_to_array(resized, dtype=np.float16) / 255.0
        images.append(resized)

        # Start making predictions only after the sequence is full
        if len(images) < SEQUENCE_LENGTH:
            continue

        X_loop = np.array([images], dtype=np.float16)
        y_loop = model.predict(X_loop, verbose=0)
        y_loop_class = np.round(y_loop).flatten().astype(int)[0]

        # Send data to wandb
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prediction = "Awake 🆙" if y_loop_class == 0 else "Sleep 💤"
        main_table.add_data(index, date, prediction, *[wandb.Image(_img) for _img in images])

        if (index-1) % SEQUENCE_LENGTH == 0:
            temp_table = create_wandb_table()
            for data in main_table.data:
                temp_table.add_data(*data)
            wandb.log({"main_table": temp_table})

        if y_loop_class == 1:
            sleep_counter.append(1)
            logger.warning("Sleep detected")
        else:
            sleep_counter.append(0)
            logger.info("No sleep detected")

        # 80% of the last 6 hours
        if sleep_counter.count(1) > 0.8 * 6 * 60:
            print("Wake up! You've been sleeping for more than 6 hours!")
            # Implement your alarm triggering mechanism here
            alarm_triggered = True
        else:
            alarm_triggered = False

wandb.finish()
