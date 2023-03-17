#!/usr/bin/python
import os
import sys

import tensorflow as tf
from tqdm import tqdm
import tensorflow_addons as tfa
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import (ConvLSTM2D, Dense, Dropout, Flatten, MaxPooling3D, TimeDistributed)
from keras.models import Sequential
from keras.preprocessing.image import image_utils, ImageDataGenerator
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

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def load_data(images_path):
    # Define data generator with augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        # horizontal_flip=True,
        # validation_split=0.2,
    )

    # Load images and labels
    X, y = [], []
    labels = []
    images = []
    for file in tqdm(list(sorted(os.listdir(images_path), key=lambda x: int(x.split('.')[0].split('-')[0])))):
        if file.endswith('.jpg'):
            logger.debug(f"Loading {file}")
            image = image_utils.load_img(os.path.join(images_path, file), keep_aspect_ratio=True, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), color_mode="grayscale")
            image = image_utils.img_to_array(image) / 255.0
            images.append(image)
            # images = np.expand_dims(image, axis=0)
            # labels = []
            if 'sleep' in file:
                labels.append(1)
            else:
                labels.append(0)

            if len(images) == SEQUENCE_LENGTH:
                X.append(np.array(images))
                y.append(labels[-1])
                images.pop(0)
                labels.pop(0)
            if len(images) > SEQUENCE_LENGTH:
                images.pop(0)
                labels.pop(0)
            # # Apply data augmentation to the images
            # for x_aug, y_aug in tqdm(datagen.flow(images, labels, batch_size=1)):
            #     X.append(x_aug)
            #     y.append(y_aug[0])
            #     if len(X) == SEQUENCE_LENGTH:
            #         break

    X = np.array(X)
    y = np.array(y)
    return X, y


if __name__ == "__main__":
    setup_logging("DEBUG")

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
