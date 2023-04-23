#!/usr/bin/env python3

from keras.preprocessing.image import ImageDataGenerator, image_utils
from collections import deque
import datetime
import os
import shlex
import shutil
import subprocess

import tqdm

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

from preprocess_data import get_image, process_data
from wandb_custom import CustomBatchEndCallback

from hyperparameters import SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, FILENAME, DATASET_NAME, create_model
from loguru import logger
import sys

def setup_logging(level = "DEBUG", show_module = False):
    """
    Setups better log format for loguru
    """
    logger.remove()  # Remove the default logger
    log_level = level
    log_fmt = u"<green>["
    log_fmt += u"{file:10.10}…:{line:<3} | " if show_module else ""
    log_fmt += u"{time:HH:mm:ss.SSS}]</green> <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(lambda x: tqdm.tqdm.write(x, end=""), level=log_level, format=log_fmt, colorize=True, backtrace=True, diagnose=True)

setup_logging("DEBUG")

def test_all_in_order(model):
    data = process_data("./data/quick")
    Xx = data[0]
    yy = data[1]
    y_out = model.predict(Xx, verbose=1)
    for i in range(len(yy)):
        y_pred = np.round(y_out[i])[0].astype(int)
        prediction = " 🆙]" if y_pred == 0 else " 💤]"
        # y_out = model.predict(np.array([Xx[0]]), verbose=0)
        # y_pred = np.round(y_out).flatten().astype(int)[0]

        if y_pred != yy[i]:
            color = "\033[91m [❌"
        else:
            color = "\033[92m [✅"

        print(color + str(prediction) + "\033[0m\t", end="")


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

    sleep_counter = list()
    images = list()
    path = "./data/new/"

    model = create_model()
    if os.path.exists(FILENAME):
        # load model from file
        # model.load_weights(FILENAME)
        model = tf.keras.models.load_model(FILENAME)
        logger.success(f"Loaded model weights from {FILENAME}")
    else:
        logger.error(f"Could not find file '{FILENAME}'")
        sys.exit(1)

    test_all_in_order(model)

    while True:
        if len(sleep_counter) >= 6 * 60:
            sleep_counter.pop(0)

        if len(images) >= SEQUENCE_LENGTH:
            images.pop(0)

        subprocess.run(shlex.split("fswebcam /tmp/aweful_tmp.jpg -d /dev/video0 -S2 -F1"),
                       check=False,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

        image = get_image("/tmp/aweful_tmp.jpg", IMAGE_HEIGHT, IMAGE_WIDTH)
        images.append(image)

        if len(images) < SEQUENCE_LENGTH:
            sleep_counter.append(0)
            logger.info(f"No sleep detected {sleep_counter.count(1)} / {len(sleep_counter)}")
            continue

        image_utils.save_img("/tmp/test.png", image)
        with tf.device("CPU"):
            y_loop = model.predict(np.array([images], dtype=np.float32), verbose=1)

        y_loop_class = np.round(y_loop).flatten().astype(int)[0]

        if y_loop_class == 1:
            sleep_counter.append(1)
            logger.warning(f"Sleep detected {sleep_counter.count(1)} / {len(sleep_counter)}: {y_loop[0][0]*100:.2f}%")
        else:
            sleep_counter.append(0)
            logger.info(f"No sleep detected {sleep_counter.count(1)} / {len(sleep_counter)}: {y_loop[0][0]*100:.2f}%")

        index += 1
        # copy the image to the data folder
        shutil.copy("/tmp/aweful_tmp.jpg", f'{path}{index}-{"awake" if y_loop_class == 0 else "sleep"}.jpg')

        if sleep_counter.count(1) > 0.8 * 6 * 60:
            print("Wake up! You've been sleeping for more than 6 hours!")
            alarm_triggered = True
        else:
            alarm_triggered = False
