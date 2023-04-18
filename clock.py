#!/usr/bin/env python3

from keras.preprocessing.image import ImageDataGenerator, image_utils
from collections import deque
import datetime
import os
import shlex
import shutil
import subprocess

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

from preprocess_data import get_image, load_individual_data, get_batches
from wandb_custom import CustomBatchEndCallback

from hyperparameters import SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, FILENAME, DATASET_NAME


def test_all_in_order(batch_size):
    for index, (X_batch, y_batch) in enumerate(get_batches(batch_size)):
        y_out = model.predict(X_batch, verbose=0)
        y_out = np.round(y_out).flatten().astype(int)

        for i in range(len(y_batch)):
            y = y_batch[i]
            y_pred = y_out[i]
            prediction = "🆙" if y_pred == 0 else "💤"

            if y != y_pred:
                color = "\033[91m"
            else:
                color = "\033[92m" if y_pred == 0 else "\033[93m"

            print(color + f"{(index * batch_size) + i:05d}:" + str(prediction) + "\033[0m", end="\t ")


from train import create_model
if __name__ == "__main__":
    index = 0
    # get the latest used index number
    for group in ["raw", "new"]:
        for filename in os.listdir("./data/" + group):
            if filename.endswith(".jpg"):
                num = int(filename.split(".")[0].split("-")[0])
                if num >= index:
                    index = num + 1

    logger.info(f"Starting with index {index}")

    sleep_counter = deque(maxlen=6 * 60)
    images = deque(maxlen=SEQUENCE_LENGTH)
    path = "./data/new/"
    input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1)

    if os.path.exists(FILENAME):
        # load model from file
        model = tf.keras.models.load_model(FILENAME)
        logger.success(f"Loaded model weights from {FILENAME}")
    else:
        logger.error(f"Could not find file '{FILENAME}'")
        sys.exit(1)

    test_all_in_order(16)
    while True:
        subprocess.run(shlex.split("fswebcam /tmp/aweful_tmp.jpg -d /dev/video0 -S2 -F1"),
                       check=False,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

        sequences = get_image("/tmp/aweful_tmp.jpg", IMAGE_HEIGHT, IMAGE_WIDTH)
        images.append(sequences)

        if len(images) < SEQUENCE_LENGTH:
            # sleep_counter.append(0)
            logger.info(f"No sleep detected {sleep_counter.count(1)} / {len(sleep_counter)}")
            continue

        X_loop = np.array([images])
        image_utils.save_img("/tmp/test.png", X_loop[0][0])
        y_loop = model.predict(X_loop, verbose=0)
        y_loop_class = np.round(y_loop).flatten().astype(int)[0]

        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prediction = "Awake 🆙" if y_loop_class == 0 else "Sleep 💤"

        if y_loop_class == 1:
            sleep_counter.append(1)
            logger.warning(f"Sleep detected {sleep_counter.count(1)} / {len(sleep_counter)}: {round(y_loop[0][0]*100, 2)}%")
        else:
            sleep_counter.append(0)
            logger.info(f"No sleep detected {sleep_counter.count(1)} / {len(sleep_counter)}: {round(y_loop[0][0]*100, 2)}%")

        index += 1
        # copy the image to the data folder
        shutil.copy("/tmp/aweful_tmp.jpg", f'{path}{index}-{"awake" if y_loop_class == 0 else "sleep"}.jpg')

        if sleep_counter.count(1) > 0.8 * 6 * 60:
            print("Wake up! You've been sleeping for more than 6 hours!")
            alarm_triggered = True
        else:
            alarm_triggered = False
