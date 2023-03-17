#!/usr/bin/python
import os
import sys

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import (ConvLSTM2D, Dense, Dropout, Flatten, MaxPooling3D, TimeDistributed)
from keras.models import Sequential
from keras.preprocessing.image import image_utils
from loguru import logger
from sklearn.model_selection import train_test_split

SEQUENCE_LENGTH = 15
IMAGE_HEIGHT = 480 // 5
IMAGE_WIDTH = 640 // 5
EPOCHS = 10
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


def create_convlstm_model():
    '''
    This function will construct the required convlstm model.
    Returns:
        model: It is the required constructed convlstm model.
    '''
    model = Sequential()

    model.add(
        ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            activation='tanh',
            recurrent_dropout=0.2,
            return_sequences=True,
            input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1),
        ))

    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(
        ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            activation='tanh',
            recurrent_dropout=0.2,
            return_sequences=True,
        ))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(
        ConvLSTM2D(
            filters=128,
            kernel_size=(3, 3),
            activation='tanh',
            recurrent_dropout=0.2,
            return_sequences=True,
        ))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    return model


def load_data(images_path):
    # Define data generator with augmentation parameters
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),])

    # Load images and labels
    dataset = tf.keras.preprocessing.image_dataset_from_directory(images_path,
                                                                  labels="inferred",
                                                                  label_mode="binary",
                                                                  class_names=["not_sleep", "sleep"],
                                                                  color_mode="grayscale",
                                                                  batch_size=1,
                                                                  image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                                  shuffle=True,
                                                                  seed=42,
                                                                  validation_split=0.2,
                                                                  subset="training")

    X, y = [], []
    for images, labels in dataset:
        for image, label in zip(images, labels):
            # Apply data augmentation to the image
            augmented_image = data_augmentation(image)
            X.append(augmented_image.numpy())
            y.append(label.numpy())

            # Break after collecting SEQUENCE_LENGTH samples
            if len(X) == SEQUENCE_LENGTH:
                break

    X = np.array(X)
    y = np.array(y)
    return X, y


if __name__ == "__main__":
    setup_logging("INFO")

    X, y = load_data("./data_new/")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    convlstm_model = create_convlstm_model()
    convlstm_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=["accuracy"])

    # train

    if not os.path.exists(FILENAME):
        convlstm_model_training_history = convlstm_model.fit(
            x=X_train,
            y=y_train,
            epochs=EPOCHS,
            batch_size=2,
            validation_data=(X_test, y_test),
            callbacks=[EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.001)],
        )
        convlstm_model.save_weights(FILENAME)

    # predict the test set
    convlstm_model.load_weights(FILENAME)
    convlstm_model_predictions = convlstm_model.predict(X)
    convlstm_model_predictions = (convlstm_model_predictions > 0.5).astype(int)

    # check and show results
    last_result = 0
    for i in range(len(convlstm_model_predictions)):
        if convlstm_model_predictions[i] != y[i]:
            logger.warning(f"This was predicted {'sleep' if convlstm_model_predictions[i] else 'not sleep'} and is actually {'sleep' if y[i] else 'not sleep'}")
        else:
            logger.success(f"{i+1}.jpg was predicted {'sleep' if convlstm_model_predictions[i] else 'not sleep'}!")
        if last_result != y[i]:
            image = image_utils.array_to_img(X[i][0])
            image.show()
            input()
            last_result = y[i]
