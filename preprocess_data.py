#!/usr/bin/env python3

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import zip_longest
from loguru import logger

import cv2
import numpy as np
from tqdm.auto import tqdm

from hyperparameters import SEQUENCE_LENGTH, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, EPOCHS, LEARNING_RATE, PATIENCE, DEBUG, FILENAME


def random_transform(image, seed):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-5, 5)
    tx = rng.integers(-10, 11)
    ty = rng.integers(-10, 11)

    # Apply rotation
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    # Apply translation
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

    image = np.expand_dims(image, axis=-1)
    return image


def get_random_crop(image, seed):
    # TODO: fix this
    rng = np.random.default_rng(seed)

    height, width = image.shape[0], image.shape[1]
    aspect_ratio = float(width) / float(height)

    if width > height:
        new_width = np.random.randint(int(width * 0.7), width)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = np.random.randint(int(height * 0.7), height)
        new_width = int(new_height * aspect_ratio)

    x = np.random.randint(0, width - new_width)
    y = np.random.randint(0, height - new_height)

    crop = image[y:y + new_height, x:x + new_width]
    resized_crop = cv2.resize(crop, (width, height))

    resized_crop = np.expand_dims(resized_crop, axis=-1)
    return resized_crop


def get_sequence(images_path):
    sequence = []
    augmented_sequence = []
    for image_path in images_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image[0:image.shape[0], 0:image.shape[1] - 21]
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = np.expand_dims(image, axis=-1)
        sequence.append(image)

    return sequence, augmented_sequence


def get_image(image_path, image_height, image_width):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image[0:image.shape[0], 0:image.shape[1] - 21]
    image = cv2.resize(image, (image_width, image_height))
    image = np.expand_dims(image, axis=-1) # Add an extra channel dimension
    return image


def process_image_sequence(image_files, images_path, image_height, image_width):
    sequence = []
    augmented_sequence = []
    labels = []
    seed = np.random.randint(0, 1000000)

    for filename in image_files:
        image = get_image(os.path.join(images_path, filename), image_height, image_width)
        label = 1 if "sleep" in filename else 0
        sequence.append(image)
        labels.append(label)

        augmented_image = random_transform(image, seed=seed)
        augmented_sequence.append(augmented_image)

    return sequence, augmented_sequence, labels[-1]


def process_data(input_path):
    X, y = [], []
    X_aug, y_aug = [], []

    # Get the list of image files
    image_files = sorted((file for file in os.listdir(input_path) if file.endswith(".jpg")),
                         key=lambda x: int(x.split(".")[0].split("-")[0]))

    # Use a ThreadPoolExecutor to process image sequences in parallel
    with ThreadPoolExecutor(max_workers=32) as executor:
        # Create the tqdm progress bar
        progress_bar = tqdm(
            total=len(image_files) - SEQUENCE_LENGTH,
            smoothing=0.1,
            desc="Processing images...",
        )

        # Submit the tasks to the executor
        futures = [
            executor.submit(process_image_sequence, image_files[i:i + SEQUENCE_LENGTH], input_path, IMAGE_HEIGHT, IMAGE_WIDTH)
            for i in range(0,
                           len(image_files) - SEQUENCE_LENGTH)]

        # Use as_completed() to process the results as they become available, and update the progress bar
        for future in as_completed(futures):
            sequence, augmented_sequence, label = future.result()
            X.append(sequence)
            y.append(label)
            X_aug.append(augmented_sequence)
            y_aug.append(label)
            progress_bar.update(1)

        progress_bar.close()

    X = np.array(X) / 255.0
    y = np.array(y)
    X_aug = np.array(X_aug) / 255.0
    y_aug = np.array(y_aug)

    X = np.concatenate((X, X_aug))
    y = np.concatenate((y, y_aug))

    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    return X, y


def load_data(key):
    loaded = np.load(os.path.join("./prep/", f"{key}.npz"))
    sequences = loaded["sequences"]
    labels = loaded["labels"]
    return sequences, labels


def save_data(key):
    sequences, labels = process_data(f"./data/{key}")

    # Save the data to disk
    logger.info(f"Saving {key} data to disk...")
    np.savez_compressed(os.path.join("./prep/", key), sequences=sequences, labels=labels)


if __name__ == "__main__":
    save_data("raw")
