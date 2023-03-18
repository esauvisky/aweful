#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import ConvLSTM2D, Dense, Dropout, Flatten, Input, MaxPooling3D, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, image_utils
from keras.preprocessing import image as image_utils
from loguru import logger

def create_model(input_shape):
    """Create a ConvLSTM model."""
    inputs = Input(shape=input_shape)
    x = ConvLSTM2D(filters=2, kernel_size=(3, 3), activation="tanh", recurrent_dropout=0.2, return_sequences=True)(inputs)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Flatten())(x)
    x = Flatten()(x)
    x = Dense(64, activation="sigmoid")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    out_model = Model(inputs=inputs, outputs=outputs)
    out_model.summary()
    return out_model

# Load the model
model = create_model((15, 48, 64, 1))
model_path = "weights-15_48_64.h5"
model.load_weights(model_path)
# if os.path.exists(FILENAME):
# model = load_model(model_path, compile=False)

# Parameters
sequence_length = 15
image_height = 48
image_width = 64
sleep_duration_limit = 6 * 60 * 60  # 6 hours in seconds
sleep_start_time = None
alarm_triggered = False

# Initialize the camera
# cap = cv2.VideoCapture(0)

# Initialize the image sequence
image_sequence = []

while True:
    # Capture an image from the webcam
    # ret, frame = cap.read()
    # if not ret:
    #     break

    # Preprocess the image
    path = "/tmp/image.jpg"
    try:
        os.system(f"fswebcam {path} -d /dev/video0 -S2 -F1")
    except:
        continue
    image = image_utils.image_utils.load_img(
        path,
        target_size=(image_height, image_width),
        color_mode="grayscale",
    )
    image = image_utils.image_utils.img_to_array(image) / 255.0

    # Add the preprocessed image to the sequence
    image_sequence.append(image)

    # If the sequence is of the required length, make a prediction
    if len(image_sequence) == sequence_length:
        X = np.expand_dims(np.array(image_sequence), axis=0)
        X = np.expand_dims(X, axis=-1)
        y_pred = model.predict(X, verbose=0)
        y_pred_class = (y_pred > 0.5).astype(int)

        # If the model predicts sleep, start the sleep timer
        if y_pred_class == 1:
            logger.warning("Sleep detected")
            if sleep_start_time is None:
                sleep_start_time = time.time()
        else:
            logger.info("No sleep detected")
            sleep_start_time = None

        # Check if the sleep duration has exceeded the limit
        if sleep_start_time is not None:
            sleep_duration = time.time() - sleep_start_time
            if sleep_duration > sleep_duration_limit and not alarm_triggered:
                # Trigger the alarm
                print("Wake up! You've been sleeping for more than 6 hours!")
                # Implement your alarm triggering mechanism here
                alarm_triggered = True
        else:
            alarm_triggered = False

        # Remove the oldest image from the sequence
        image_sequence.pop(0)

    # Display the image
    # cv2.imshow("Sleep Detection", frame)
    # key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break from the loop
    # if key == ord("q"):
    #     break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
