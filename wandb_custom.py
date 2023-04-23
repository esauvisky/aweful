import datetime
import random
import numpy as np
import tensorflow as tf

import wandb
from keras.callbacks import Callback
from hyperparameters import SEQUENCE_LENGTH, BATCH_SIZE


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
        table = wandb.Table(allow_mixed_types=True, columns=["Video", "Prediction", "Actual", "Prediction Weight"])

        def process_predictions(result):
            y = np.round(result).flatten().astype(int)[0]
            label = "Sleep" if y == 1 else "Awake"
            return np.array([result, label])

        random_ixs = np.random.permutation(len(self.X_wandb))[:64 // BATCH_SIZE * BATCH_SIZE]
        results = self.model.predict(np.array(self.X_wandb)[random_ixs])

        for c, ix in enumerate(random_ixs):
            value, label = tf.numpy_function(process_predictions, results[c], Tout=tf.uint8)
            video_raw = np.array(self.X_wandb[ix]) * 255
            # Convert the array to the uint8 data type
            video_uint8 = video_raw.astype(np.uint8)
            # Remove the extra dimension (8, 240, 320)
            video_squeezed = np.squeeze(video_uint8, axis=-1)
            # Repeat the channel dimension 3 times to simulate an RGB image (8, 3, 240, 320)
            video_rgb = np.repeat(video_squeezed[:, np.newaxis, :, :], 3, axis=1)
            video = wandb.Video(video_rgb, fps=4)

            actual_lbl = 'Sleep' if self.y_wandb[ix].numpy() == 1 else 'Awake'
            table.add_data(video, label, actual_lbl, value)

        wandb.log({"Prediction Table": table}, commit=False)
