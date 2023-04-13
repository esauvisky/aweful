import datetime
import os
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split


SEQUENCE_LENGTH = 16
BATCH_SIZE = 4

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

def create_wandb_images_table():
    columns = ['Index', 'Date', 'Prediction']
    for s in range(SEQUENCE_LENGTH):
        columns.append(f'Sample {s + 1}')
    return wandb.Table(columns=columns, allow_mixed_types=True)


def create_wandb_predictions_table():
    columns = ['Index', 'Date', 'Prediction']
    return wandb.Table(columns=columns, allow_mixed_types=True)


class CustomBatchEndCallback(Callback):
    def __init__(self, X, y, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.test_table = create_wandb_images_table()

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        if batch % random.randrange(8, 15) == 0:
            # Get the X value (array of images) for the first sample in the batch
            X_step = self.X[batch * BATCH_SIZE]
            y_step = "Sleep" if self.y[batch * BATCH_SIZE] == 1 else "Awake"

            # create a table with each image of the sequence
            images = []
            for sequence_ix, _img in enumerate(X_step):
                # image is the same as self.x_train[batch_ix * BATCH_SIZE + ix][0]
                img = wandb.Image(_img, caption=f"Image {batch * BATCH_SIZE + sequence_ix} - Label: {y_step}")
                images.append(img)

            # Adds the row to the table with the actual index of the first image of the sequence,
            # the original label y, and the images
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.test_table.add_data(batch * BATCH_SIZE, date, y_step, *images)
            print(f" | Samples {batch * BATCH_SIZE}-{batch * BATCH_SIZE + SEQUENCE_LENGTH - 1} - Label: {y_step})'")

    def on_epoch_end(self, epoch, logs=None):
        wandb.log({"data": self.test_table}, commit=True)
        self.test_table = create_wandb_images_table()
        super().on_epoch_end(epoch, logs)

