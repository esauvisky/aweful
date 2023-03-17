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
            filters=4,
            kernel_size=(3, 3),
            activation='tanh',
            recurrent_dropout=0.2,
            return_sequences=True,
            input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1),
        ))

    # model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    # model.add(TimeDistributed(Dropout(0.2)))
    # model.add(
    #     ConvLSTM2D(
    #         filters=8,
    #         kernel_size=(3, 3),
    #         activation='tanh',
    #         recurrent_dropout=0.2,
    #         return_sequences=True,
    #     ))
    # model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    # model.add(TimeDistributed(Dropout(0.2)))
    # model.add(
    #     ConvLSTM2D(
    #         filters=16,
    #         kernel_size=(3, 3),
    #         activation='tanh',
    #         recurrent_dropout=0.2,
    #         return_sequences=True,
    #     ))
    # model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    return model


def load_data(images_path):
    X, y = [], []
    images = []
    labels = []
    for file in sorted(os.listdir(images_path), key=lambda x: int(x.split('.')[0].split('-')[0])):
        if file.endswith('.jpg'):
            logger.debug(f"Loading {file}")
            image = image_utils.load_img(os.path.join(images_path, file), keep_aspect_ratio=True, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), color_mode="grayscale")
            image = image_utils.img_to_array(image) / 255.0
            images.append(image)
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
    X = np.array(X)
    y = np.array(y)
    return X, y


if __name__ == "__main__":
    setup_logging("INFO")

    X, y = load_data("./data/")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True)

    convlstm_model = create_convlstm_model()
    convlstm_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=["accuracy"])

    # train

    if not os.path.exists(FILENAME):
        convlstm_model_training_history = convlstm_model.fit(
            x=X_train,
            y=y_train,
            epochs=EPOCHS,
            batch_size=8,
            validation_data=(X_test, y_test),
            callbacks=[EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.02)],
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
