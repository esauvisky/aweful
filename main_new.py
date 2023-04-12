#!/usr/bin/env python3

import datetime
import os

from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os
import random
import sys

import cv2
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import ConvLSTM2D, Dense, Flatten, Input, MaxPooling3D
from keras.models import Model
from keras.optimizers import Adam
from loguru import logger
from wandb.keras import WandbCallback

import wandb
from load_data_old import load_data

# Define default hyperparameters
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT = 480 // 10
IMAGE_WIDTH = 640 // 10
BATCH_SIZE = 2
EPOCHS = 2
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
    x = Flatten()(x)
    x = Dense(4, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    out_model = Model(inputs=inputs, outputs=outputs)
    out_model.summary()
    return out_model


def process_image_sequence(images):
    def process_image_file(image):
        # Load the image
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        # Crop the bottom 21 pixels
        cropped_image = image[:-21, :]

        # Resize the image to the target dimensions
        resized_image = cv2.resize(cropped_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # Normalize the image
        normalized_image = resized_image / 255.0

        # Add a new dimension for the single channel
        normalized_image = normalized_image[..., np.newaxis]

        return normalized_image

    return [process_image_file(file) for file in images]


def get_generator(images_path, split_type):
    image_files = [f for f in os.listdir(images_path)]
    image_files = sorted(image_files, key=lambda x: int(x.split(".")[0].split("-")[0]))
    if split_type == "train":
        image_files = image_files[:int(len(image_files) * 0.8)]
    elif split_type == "val":
        image_files = image_files[int(len(image_files) * 0.8):]

    def generator():
        while True:
            batch_images = []
            batch_labels = []
            for i in range(len(image_files) - SEQUENCE_LENGTH):
                label = tf.constant(1, dtype=tf.int8) if "sleep" in image_files[-1] else tf.constant(0, dtype=tf.int8)
                batch_images.append(
                    process_image_sequence([os.path.join(images_path, f) for f in image_files[i:i + SEQUENCE_LENGTH]]))
                batch_labels.append(label)

                if len(batch_images) == BATCH_SIZE:
                    yield batch_images, batch_labels
                    batch_images = []
                    batch_labels = []

    generator = tf.data.Dataset.from_generator(generator=generator,
                                               output_types=(tf.float16, tf.int8),
                                               output_shapes=(
                                                   (BATCH_SIZE, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1),
                                                   (BATCH_SIZE,),
                                               )).prefetch(tf.data.experimental.AUTOTUNE)

    step_count = len(image_files) // BATCH_SIZE

    return generator, step_count


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        sys.argv.pop(1)
        routine = "train"
    elif len(sys.argv) >= 2 and sys.argv[1] == "check":
        routine = "check"
    else:
        routine = "clock"

    setup_logging("DEBUG" if DEBUG else "INFO")
    logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    # Create the model
    input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    model = create_model(input_shape)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Start wandb
    wandb.init(project=f"aweful-{routine}",
               config={
                   "optimizer": "adam",
                   "loss": "binary_crossentropy",
                   "metric": "accuracy",
                   "epoch": EPOCHS,
                   "batch_size": BATCH_SIZE,})

    if routine == "train":
        train_dataset, steps_per_epoch = get_generator("./data/0/", "train")
        val_dataset, validation_steps = get_generator("./data/0/", "val")

        callbacks = [
            WandbCallback(log_weights=True,
                          log_evaluation=True,
                          log_batch_frequency=10,
                          log_evaluation_frequency=10,
                          log_weights_frequency=10,
                          save_model=False),
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ModelCheckpoint(FILENAME, monitor="val_loss", save_best_only=True, verbose=1)]

        model.fit(train_dataset,
                  epochs=EPOCHS,
                  callbacks=callbacks,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=val_dataset,
                  validation_steps=validation_steps)

        # Save the model weights
        model.save_weights(FILENAME)
        logger.info("Saved model weights")

    elif os.path.exists(FILENAME):
        model.load_weights(FILENAME)
        logger.success(f"Loaded model weights from {FILENAME}")

    else:
        logger.error(f"Could not find file '{FILENAME}'")
        if routine != "train": return
        return

    with tf.device("GPU"):
        X, y = load_data("./test/", SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

        for i in range(0, len(X), BATCH_SIZE):
            X_predict = X[i:i + BATCH_SIZE]
            y_predict = y[i:i + BATCH_SIZE]
            y_out = model.predict(X_predict, verbose=0)
            y_out = np.round(y_out).flatten().astype(int)
            for n, cat in enumerate(y_out):
                prediction = "🆙" if cat == 0 else "💤"
                if y_predict[n] != cat: color = "\033[91m"
                else: color = "\033[92m" if cat == 0 else "\033[93m"
                print(color + f"{i:05d}:" + str(prediction) + "\033[0m", end=" | ")

    # if routine == "clock":
    #     index = 0
    #     # get the latest used index number
    #     for group in os.listdir("./data"):
    #         for filename in os.listdir("./data/" + group):
    #             if filename.endswith(".jpg"):
    #                 num = int(filename.split(".")[0].split("-")[0])
    #                 if num >= index:
    #                     index = num + 1

    #     sleep_counter = deque(maxlen=6 * 60)
    #     images = deque(maxlen=SEQUENCE_LENGTH)
    #     path = f"./data/{int(group) + 1}/"

    #     while True:
    #         subprocess.run(shlex.split(f"fswebcam /tmp/aweful_tmp.jpg -d /dev/video0 -S2 -F1"),
    #                        check=False,
    #                        stdout=subprocess.DEVNULL,
    #                        stderr=subprocess.DEVNULL)

    #         image = get_image(f"/tmp/aweful_tmp.jpg", IMAGE_HEIGHT, IMAGE_WIDTH)
    #         images.append(image)
    #         # # image = image_utils.load_img(f"{path}{index}.jpg", color_mode="grayscale")
    #         # # image = image_utils.img_to_array(image)
    #         # # image = image[:-21, :]

    #         # resized = image_utils.array_to_img(image).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    #         # resized = image_utils.img_to_array(resized, dtype=np.float32) / 255.0
    #         # images.append(resized)

    #         if len(images) < SEQUENCE_LENGTH:
    #             # sleep_counter.append(0)
    #             # logger.info(f"No sleep detected {sleep_counter.count(1)} / {len(sleep_counter)}")
    #             continue

    #         X_loop = np.array([images], dtype=np.float32)
    #         y_loop = model.predict(X_loop, verbose=0)
    #         y_loop_class = np.round(y_loop).flatten().astype(int)[0]

    #         date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         prediction = "Awake 🆙" if y_loop_class == 0 else "Sleep 💤"

    #         if y_loop_class == 1:
    #             sleep_counter.append(1)
    #             logger.warning(f"Sleep detected {sleep_counter.count(1)} / {len(sleep_counter)}")
    #         else:
    #             sleep_counter.append(0)
    #             logger.info(f"No sleep detected {sleep_counter.count(1)} / {len(sleep_counter)}")

    #         index += 1
    #         # copy the image to the data folder
    #         shutil.copy("/tmp/aweful_tmp.jpg", f'{path}{index}-{"awake" if y_loop_class == 0 else "sleep"}.jpg')

    #         if sleep_counter.count(1) > 0.8 * 6 * 60:
    #             print("Wake up! You've been sleeping for more than 6 hours!")
    #             alarm_triggered = True
    #         else:
    #             alarm_triggered = False
    wandb.finish()


if __name__ == "__main__":
    main()
