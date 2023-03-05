import os
import sys

import numpy as np
from loguru import logger
import numpy as np
import tensorflow as tf
from keras.layers import LSTM, Conv2D, Dense, Flatten, MaxPooling2D, Reshape, ConvLSTM2D, MaxPooling3D, TimeDistributed, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import image_utils
from tensorflow import keras
from sklearn.model_selection import train_test_split

SEQUENCE_LENGTH = 30
IMAGE_HEIGHT=64
IMAGE_WIDTH=48

def setup_logging(level = "DEBUG", show_module = False):
    """
    Setups better log format for loguru
    """
    logger.remove(0)  # Remove the default logger
    log_level = level
    log_fmt = u"<green>["
    log_fmt += u"{file:10.10}…:{line:<3} | " if show_module else ""
    log_fmt += u"{time:HH:mm:ss.SSS}]</green> <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, level=log_level, format=log_fmt, colorize=True, backtrace=True, diagnose=True)

setup_logging("DEBUG")

def create_convlstm_model():
    '''
    This function will construct the required convlstm model.
    Returns:
        model: It is the required constructed convlstm model.
    '''

    # We will use a Sequential model for model construction
    model = Sequential()

    # Define the Model Architecture.
    ########################################################################################################################

    model.add(ConvLSTM2D(filters = 4, kernel_size = (3, 3), activation = 'tanh',data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True, input_shape = (SEQUENCE_LENGTH,
                                                                                      IMAGE_HEIGHT, IMAGE_WIDTH, 1)))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(ConvLSTM2D(filters = 8, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last", recurrent_dropout=0.2, return_sequences=True))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    # model.add(TimeDistributed(Dropout(0.2)))

    # model.add(ConvLSTM2D(filters = 14, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last", recurrent_dropout=0.2, return_sequences=True))

    # model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    # model.add(TimeDistributed(Dropout(0.2)))

    # model.add(ConvLSTM2D(filters = 16, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last", recurrent_dropout=0.2, return_sequences=True))

    # model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))

    model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(1, activation = "softmax"))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    return model


X, y = [], []
images_path = '../alarm-clock/'
images = []
labels = []
for file in sorted(os.listdir(images_path), key=lambda x: int(x.split('.')[0].split('-')[0])):
    if file.endswith('.jpg'):
        logger.debug(f"Loading {file}")
        image = image_utils.load_img(os.path.join(images_path, file), keep_aspect_ratio=True, target_size=(IMAGE_HEIGHT,IMAGE_WIDTH), color_mode="grayscale")
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


convlstm_model = create_convlstm_model()
convlstm_model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)
# convlstm_model_training_history = convlstm_model.fit(x = X_train, y = y_train, epochs = 50, batch_size = 4, shuffle = True, validation_split = 0.2, callbacks = [early_stopping_callback])
# convlstm_model.save('convlstm_model.h5')
# convlstm_model.save_weights('convlstm_model_weights.h5')
convlstm_model.load_weights('convlstm_model_weights.h5')

# predict the test set
convlstm_model_predictions = convlstm_model.predict(X)
convlstm_model_predictions = (convlstm_model_predictions > 0.5).astype(int)
for i in range(len(convlstm_model_predictions)):
    if convlstm_model_predictions[i] == 1:
        print(images_path, i+1, ".jpg was predicted sleep and is actually ", y[i])
    else:
        print(images_path, i+1, ".jpg was predicted not sleep and is actually ", y[i])
# print(convlstm_model_predictions)
