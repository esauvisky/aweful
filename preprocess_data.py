#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from loguru import logger

import cv2
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from keras.preprocessing.image import ImageDataGenerator, image_utils

from hyperparameters import SEQUENCE_LENGTH, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, EPOCHS, LEARNING_RATE, PATIENCE, DEBUG, FILENAME

SEED = np.random.randint(0, 1000000)
datagen = ImageDataGenerator(height_shift_range=(-0.18, 0.10),
                             width_shift_range=0.1,
                             zoom_range=0.05,
                             fill_mode='reflect')


def simulate_panning(images, seed):
    # rotation_range=4,
    # brightness_range=(0.5, 1),
    # shear_range=0.05,

    # Add an extra dimension for the batch size
    images = np.array(images)

    # Apply the data augmentation
    augmented_image_iterator = datagen.flow(images, batch_size=len(images), seed=seed)

    # Get the augmented images from the iterator
    augmented_images = next(augmented_image_iterator).astype(np.uint8)

    return augmented_images


def show_image(image_or_sequence):
    if len(np.array(image_or_sequence).shape) == 4:
        for image in image_or_sequence:
            show_image(image)
    else:
        # Read the input image
        # image = cv2.imread(image_or_sequence, cv2.IMREAD_GRAYSCALE)

        # Simulate panning
        panned_image = simulate_panning(image_or_sequence)

        # Display the original and panned images
        cv2.imshow('Original Image', image_or_sequence)
        cv2.imshow('Panned Image', panned_image)

        # Wait for a key press and close the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
    global SEED
    sequence = []
    augmented_sequence = []
    labels = []

    for filename in image_files:
        image = get_image(os.path.join(images_path, filename), image_height, image_width)
        label = 1 if "sleep" in filename else 0
        sequence.append(image)
        labels.append(label)

    augmented_sequence = simulate_panning(sequence, SEED)

    return sequence, augmented_sequence, labels[-1]


def process_data(input_path):
    global SEED
    X, X_aug, y = [], [], []

    # Get the list of image files
    image_files = sorted((file for file in os.listdir(input_path) if file.endswith(".jpg")),
                         key=lambda x: int(x.split(".")[0].split("-")[0]))

    minority_image_files = [file for file in image_files if "sleep" in file]
    majority_image_files = [file for file in image_files if "sleep" not in file]

    majority_class_count = len(majority_image_files)
    minority_class_count = len(minority_image_files)
    oversampling_factor = majority_class_count // minority_class_count

    oversampled_sequences = []
    minority_count = 0
    majority_count = 0
    for i in range(len(image_files) - SEQUENCE_LENGTH):
        sequence = image_files[i:i + SEQUENCE_LENGTH]
        if "sleep" in sequence[-1]:
            for _ in range(oversampling_factor // 3):
                minority_count += 1
                oversampled_sequences.append(sequence)
        elif random.random() < 0.5:
            majority_count += 1
            oversampled_sequences.append(sequence)

    # remove the final sequences until we have a multiple of the batch size
    oversampled_sequences = oversampled_sequences[:len(oversampled_sequences) - (len(oversampled_sequences) % BATCH_SIZE)]

    logger.info(f"Oversampled sequences: {len(oversampled_sequences)}")
    logger.info(f"Minority count: {minority_count}")
    logger.info(f"Majority count: {majority_count}")

    # Use a ThreadPoolExecutor to process image sequences in parallel
    with ThreadPoolExecutor(max_workers=24) as executor:
        # Create the tqdm progress bar
        progress_bar = tqdm(
            total=len(oversampled_sequences) - SEQUENCE_LENGTH,
            smoothing=0.1,
            desc="Processing images...",
        )

        # Submit the tasks to the executor
        offset = np.random.randint(0, 200)
        # Submit the tasks to the executor
        futures = [
            executor.submit(process_image_sequence, oversampled_sequences[i], input_path, IMAGE_HEIGHT, IMAGE_WIDTH)
            for i in range(0,
                           len(oversampled_sequences) - SEQUENCE_LENGTH)]

        # Use as_completed() to process the results as they become available, and update the progress bar
        for future in as_completed(futures):
            sequence, augmented_sequence, label = future.result()
            X.append(sequence)
            X_aug.append(augmented_sequence)
            y.append(label)

            if progress_bar.n % (500+offset) == 0:
                logger.info("Switching seed for augmentation")
                SEED = np.random.randint(0, 1000000)

            progress_bar.update(1)

        progress_bar.close()

    # logger.info(f"X_aug shape: {X_aug.shape}")
    X = np.concatenate((X, X_aug))
    y = np.concatenate((y, y))
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    return X, y


def save_individual_data(key, sequences, labels):
    output_dir = os.path.join("./prep/data", key)
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(len(sequences)), desc="Saving data..."):
        sequence = sequences[i]
        label = labels[i]

        # Save the sequence and label as individual files
        np.savez_compressed(os.path.join(output_dir, f"sequence_{i}.npz"), sequence=sequence)
        np.save(os.path.join(output_dir, f"label_{i}.npy"), label)


def load_individual_data(key):
    input_dir = os.path.join("./prep/", key)
    num_files = len(os.listdir(input_dir)) // 2

    # Initialize empty arrays for sequences and labels
    sequences = []
    labels = []

    # Iterate over the files and load the data
    for i in tqdm(range(num_files), desc="Loading data..."):
        sequence_file = os.path.join(input_dir, f"sequence_{i}.npz")
        sequences.append(np.load(sequence_file)["sequence"])

    for i in tqdm(range(num_files), desc="Loading labels..."):
        label_file = os.path.join(input_dir, f"label_{i}.npy")
        labels.append(np.load(label_file))

    sequences = np.array(sequences, dtype=np.float16)
    labels = np.array(labels, dtype=np.int8)
    return sequences, labels


def save_single_data(sequence, label, index, key):
    output_dir = os.path.join("./prep/", key)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sequence_file = os.path.join(output_dir, f"sequence_{index}.npz")
    label_file = os.path.join(output_dir, f"label_{index}.npy")

    np.savez_compressed(sequence_file, sequence=sequence)
    np.save(label_file, label)


def load_data(key):
    loaded = np.load(os.path.join("./prep/", f"{key}.npz"))
    sequences = loaded["sequences"]
    labels = loaded["labels"]
    return sequences, labels


def save_data(key):
    sequences, labels = process_data(f"./data/{key}")

    with ThreadPoolExecutor(max_workers=24) as executor:
        save_futures = [
            executor.submit(save_single_data, sequences[i], labels[i], i, key) for i in range(len(sequences))]

        # Show progress using tqdm
        progress_bar = tqdm(
            total=len(save_futures),
            smoothing=0.1,
            desc="Saving data...",
        )

        for future in as_completed(save_futures):
            progress_bar.update(1)

        progress_bar.close()


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"Num GPUs Available: {len(gpus)}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    save_data("raw")
