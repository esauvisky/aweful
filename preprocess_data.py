#!/usr/bin/env python3

import os
import re

import wandb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import random
from loguru import logger

import cv2
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from keras.preprocessing.image import ImageDataGenerator, image_utils

from hyperparameters import SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, THREADS, DATASET_NAME

from skimage import exposure
SEED = np.random.randint(0, 1000000)


def set_seed(seed=0):
    global SEED
    SEED = seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISM'] = "1"
    os.environ['PYTHONHASHSEED'] = str(seed)


def standardize_inplace(image):
    mean = np.mean(image)
    std = np.std(image)
    image -= mean
    image /= std
    return image


def simulate_panning(images):
    global SEED
    datagen = ImageDataGenerator(height_shift_range=(-0.1, 0.5),
                                 width_shift_range=(-0.5, 0.5),
                                 zoom_range=0.05,
                                 fill_mode='reflect')

    # Add an extra dimension for the batch size
    images = np.array(images)
    transformation_matrix = datagen.get_random_transform(images.shape[1:], SEED)

    augmented_images = [datagen.apply_transform(image, transformation_matrix) for image in images]

    return augmented_images


def show_image(image):
    # Read the input image
    # image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # if image.dtype != np.int32:
    #     image = image.astype(np.int32)

    # Remove the extra dimension
    image = np.squeeze(image, axis=-1)

    # Display the original and panned images
    cv2.imshow('Original Image', image)
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
    translation_matrix = np.float16([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

    image = np.expand_dims(image, axis=-1)
    return image


def get_image(image_path, image_height, image_width):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image[0:image.shape[0] - 21, 0:image.shape[1]]
    image = cv2.resize(image, (image_width, image_height))
    image = np.expand_dims(image, axis=-1) # Add an extra channel dimension
    return image


def process_image_sequence(image_files, image_height, image_width):
    global SEED
    sequence = []
    augmented_sequence = []
    labels = []

    for filename in image_files:
        image = get_image(filename, image_height, image_width)
        label = 1 if "sleep" in filename else 0
        sequence.append(image)
        labels.append(label)

    augmented_sequence = simulate_panning(sequence)

    return sequence, augmented_sequence, labels[-1]


def balance_dataset(X, X_aug, y):

    X_balanced, y_balanced = [], []
    minority_indices = [i for i, label in enumerate(y) if label == 1]
    majority_indices = [i for i, label in enumerate(y) if label == 0]

    # Determine the number of majority samples to remove
    num_to_remove = len(majority_indices) - len(minority_indices)

    # Add augmented minority data if the majority class has more than double the samples
    if num_to_remove > len(minority_indices):
        num_to_add = min(num_to_remove - len(minority_indices), len(minority_indices))
        extra_minority_indices = random.sample(minority_indices, num_to_add)
        for index in extra_minority_indices:
            X.append(X_aug[index])
            y.append(y[index])
        minority_indices += extra_minority_indices

        # Update majority_indices and num_to_remove after adding elements to X and y
        majority_indices = [i for i, label in enumerate(y) if label == 0]
        num_to_remove = len(majority_indices) - len(minority_indices)

    # Randomly remove samples from the majority class
    majority_indices = random.sample(majority_indices, len(majority_indices) - num_to_remove)

    # Combine minority and majority indices
    balanced_indices = minority_indices + majority_indices
    # random.shuffle(balanced_indices)

    for index in balanced_indices:
        X_balanced.append(X[index])
        y_balanced.append(y[index])

    return X_balanced, y_balanced


def process_data(input_dir):
    global SEED
    X, X_aug, y = [], [], []

    def custom_sort(file):
        numbers = [int(x) for x in re.findall(r'\d+', file)]
        return numbers

    image_files = sorted((os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".jpg")),
                         key=custom_sort)
    sequences = [image_files[i:i + SEQUENCE_LENGTH] for i in range(len(image_files) - SEQUENCE_LENGTH)]

    # Use a ThreadPoolExecutor to process image sequences in parallel
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        # Create the tqdm progress bar
        progress_bar = tqdm(total=len(sequences), smoothing=0.1, desc="Processing images...", position=0, leave=True)

        offset = np.random.randint(0, 500)
        futures = [executor.submit(process_image_sequence, s, IMAGE_HEIGHT, IMAGE_WIDTH) for s in sequences]

        # Use as_completed() to process the results as they become available, and update the progress bar
        for future in as_completed(futures):
            sequence, augmented_sequence, label = future.result()
            X.append(sequence)
            X_aug.append(augmented_sequence)
            y.append(label)

            if progress_bar.n % (500+offset) == 0:
                logger.info("Switching seed for augmentation")
                set_seed(np.random.randint(0, 1000000))

            progress_bar.update(1)

        progress_bar.close()

    logger.info(f"X_aug size: {len(X_aug)} | X Actual: {len(X)} | Total sequences: {len(sequences) - SEQUENCE_LENGTH}")
    X, y = balance_dataset(X, X_aug, y)
    logger.info(f"X_balanced: {len(X)} y size: {len(y)} | Classes: {np.unique(y)} | Counts: {np.bincount(y)}")

    return X, y


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

    sequences = np.array(sequences, dtype=np.uint8)
    labels = np.array(labels, dtype=np.uint8)
    return sequences, labels


def load_npz_by_idx(input_dir, idx):
    sequence_file = os.path.join(input_dir, f"sequence_{idx}.npz")
    sequence = np.load(sequence_file)["sequence"]

    label_file = os.path.join(input_dir, f"label_{idx}.npy")
    label = np.load(label_file)

    return sequence, label


def get_batches(batch_size, random=False, split_ratio=1.0):
    input_dir = os.path.join("./prep/", DATASET_NAME)
    num_files = len(os.listdir(input_dir)) // 2
    batch_X, batch_y = [], []
    if random:
        indices = np.random.permutation(range(num_files))[0:int(split_ratio * num_files)]
    else:
        indices = list(range(num_files))

    for idx in indices:
        sequence_file = os.path.join(input_dir, f"sequence_{idx}.npz")
        sequence = np.load(sequence_file)["sequence"]

        label_file = os.path.join(input_dir, f"label_{idx}.npy")
        label = np.load(label_file)

        batch_X.append(sequence)
        batch_y.append(label)

        if len(batch_X) == batch_size:
            yield np.array(batch_X).astype(np.float32), np.array(batch_y).astype(np.float32)
            batch_X, batch_y = [], []


def get_sequences(random=False, split_ratio=1.0):
    input_dir = os.path.join("./prep/", DATASET_NAME)
    num_files = len(os.listdir(input_dir)) // 2
    if random:
        indices = np.random.permutation(range(num_files))[0:int(split_ratio * num_files)]
    else:
        indices = list(range(num_files))[0:int(split_ratio * num_files)]

    for idx in indices:
        sequence_file = os.path.join(input_dir, f"sequence_{idx}.npz")
        sequence = np.load(sequence_file)["sequence"].astype(np.float32)

        label_file = os.path.join(input_dir, f"label_{idx}.npy")
        label = np.load(label_file).astype(np.float32)

        yield sequence, label


def save_single_data(sequence, label, index, key):
    output_dir = os.path.join("./prep/", key)

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    sequence_file = os.path.join(output_dir, f"sequence_{index}.npz")
    label_file = os.path.join(output_dir, f"label_{index}.npy")

    np.savez_compressed(sequence_file, sequence=sequence)
    np.save(label_file, label)


def save_data(key):
    sequences, labels = process_data("./data/raw")

    table = wandb.Table(columns=["label", "video"])
    for ix in random.sample(range(0, len(sequences)), 10):

        def get_video(seq):
            # Convert the array to the uint8 data type
            # video_uint8 = np.array(seq).astype(np.uint8)
            # Remove the extra dimension (8, 240, 320)
            video_squeezed = np.squeeze(seq, axis=-1)
            # Repeat the channel dimension 3 times to simulate an RGB image (8, 3, 240, 320)
            video_rgb = np.repeat(video_squeezed[:, np.newaxis, :, :], 3, axis=1)
            return wandb.Video(video_rgb, fps=4)

        table.add_data(labels[ix], get_video(sequences[ix]))
        # table.add_data(labels[ix], get_video(augmented_sequences[ix]))

    wandb.log({key: table})

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        save_futures = [
            executor.submit(save_single_data, sequences[i], labels[i], i, key) for i in range(len(sequences))]
        # Show progress using tqdm
        progress_bar = tqdm(
            total=len(save_futures),
            smoothing=0.1,
            desc="Saving data...",
        )

        for _ in as_completed(save_futures):
            progress_bar.update(1)

        progress_bar.close()


if __name__ == "__main__":
    wandb.init(project="aweful-preprocess")
    # gpus = tf.config.list_physical_devices('GPU')
    # logger.info(f"Num GPUs Available: {len(gpus)}")
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    save_data(DATASET_NAME)
    wandb.finish()
