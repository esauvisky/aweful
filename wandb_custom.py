import datetime
import random
import numpy as np
import tensorflow as tf

import wandb
from keras.callbacks import Callback
from hyperparameters import SEQUENCE_LENGTH, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, EPOCHS, LEARNING_RATE, PATIENCE, DEBUG, FILENAME


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
            X_step = self.model(np.array([self.X_wandb[0]]))
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


class CustomEpochEndWandbCallback(Callback):
    def __init__(self, X_wandb, y_wandb):
        super().__init__()
        self.X_wandb = X_wandb
        self.y_wandb = y_wandb

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        table = wandb.Table(columns=["Video", "Prediction", "Actual", "Prediction Weight"])

        def process_predictions(result):
            label = "Awake" if np.argmax(result) == 0 else "Sleep"
            return np.array([result, label])

        data = []
        for n, sequence in enumerate(self.X_wandb):
            prediction = self.model.predict([np.array([sequence])])[0]
            data.append(tf.numpy_function(process_predictions, prediction, Tout=tf.float32))

        for n, ((value, prediction), original) in enumerate(zip(data, self.y_wandb)):
            video_raw = self.X_wandb[n]
            # Convert the array to the uint8 data type
            video_uint8 = (video_raw * 255).astype(np.uint8)
            # Remove the extra dimension (8, 240, 320)
            video_squeezed = np.squeeze(video_uint8, axis=-1)
            # Repeat the channel dimension 3 times to simulate an RGB image (8, 3, 240, 320)
            video_rgb = np.repeat(video_squeezed[:, np.newaxis, :, :], 3, axis=1)
            video = wandb.Video(video_rgb, fps=4)

            table.add_data(video, prediction, original, value)

        wandb.log({"Prediction Table": table}, commit=False)
