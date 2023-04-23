#!/usr/bin/env python3

import asyncio
from collections import Counter
import os
import re

import wandb

from image_display import display_image_grid_from_arrays
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import random
from loguru import logger

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from keras.preprocessing.image import ImageDataGenerator, image_utils

from hyperparameters import BATCH_SIZE, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, THREADS, DATASET_NAME
from skimage.metrics import structural_similarity as ssim
from skimage import exposure


def set_seed(seed=0):
    global SEED
    SEED = seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISM'] = "1"
    os.environ['PYTHONHASHSEED'] = str(seed)


def show_image(image):
    # Remove the extra dimension
    image = np.squeeze(image, axis=-1)

    # Display the original and panned images
    cv2.imshow('Original Image', image)
    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def simulate_panning(images):
    global SEED
    datagen = ImageDataGenerator(height_shift_range=0.05, width_shift_range=0.05, fill_mode='reflect')

    # Add an extra dimension for the batch size
    images = np.array(images)
    transformation_matrix = datagen.get_random_transform(images.shape[1:], SEED)

    augmented_images = [datagen.apply_transform(image, transformation_matrix) for image in images]

    return augmented_images


def show_kmeans_img(img_vect):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, centroids = cv2.kmeans(img_vect, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centroids = np.uint8(centroids)
    img_kmeans = centroids[label.flatten()]
    img_kmeans = img_kmeans.reshape((1680, 1080))
    return img_kmeans
    # plt.title("KMeans Segmentation")
    # plt.imshow(cv2.cvtColor(img_kmeans, cv2.COLOR_GRAY2RGB))


def get_image(image_path, image_height, image_width):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image[0:image.shape[0] - 21, 0:image.shape[1]]
    image = cv2.resize(image, (image_width, image_height))
    image = np.expand_dims(image, axis=-1) # Add an extra channel dimension
    return (image / 255.0).astype(np.float16)


def is_almost_black(image, threshold=10):
    """Check if the average pixel intensity of an image is below a threshold value."""
    # Remove the extra dimension
    image = np.squeeze(image, axis=-1)
    avg_intensity = np.mean((image * 255).astype(np.uint8))
    return avg_intensity < threshold


def similarity(image1, image2):
    # Remove the extra dimensions
    image1 = (np.squeeze(image1, axis=-1) * 255).astype(np.uint8)
    image2 = (np.squeeze(image2, axis=-1) * 255).astype(np.uint8)
    similarity = ssim(image1, image2)
    return similarity

def balance_classes(sequences, labels):
    # Count the occurrences of each label
    label_counts = Counter(labels)

    # Find the major and minor categories
    major_category = max(label_counts, key=label_counts.get)
    minor_category = min(label_counts, key=label_counts.get)

    major_sequences = [seq for seq, label in zip(sequences, labels) if label == major_category]
    minor_sequences = [seq for seq, label in zip(sequences, labels) if label == minor_category]

    # Calculate the value of k to balance the number of items in each category
    k = label_counts[major_category] // label_counts[minor_category]

    # Generate k augmented items for each minor category item
    transformed_removed_major_sequences = []
    augmented_minor_sequences = []
    for seq in tqdm(minor_sequences, leave=False, position=0):
        for _ in range(k):
            augmented_seq = simulate_panning(seq)
            augmented_minor_sequences.append(augmented_seq)
            # removes one sequence from original major sequence and
            # pans it the same way than the minor sequence image above.
            major_seq = major_sequences.pop(0)
            transformed_seq = simulate_panning(major_seq)
            transformed_removed_major_sequences.append(transformed_seq)

    # Combine the major sequences, augmented minor sequences, and transformed removed major sequences
    balanced_sequences = major_sequences + augmented_minor_sequences + transformed_removed_major_sequences
    balanced_labels = (
        [major_category] * len(major_sequences)
        + [minor_category] * len(augmented_minor_sequences)
        + [major_category] * len(transformed_removed_major_sequences)
    )
    # pairs = list(zip(balanced_sequences, balanced_labels))
    # random.shuffle(pairs)
    # balanced_sequences, balanced_labels = zip(*pairs)
    return balanced_sequences, balanced_labels

def process_data(input_dir):
    def custom_sort(file):
        numbers = [int(x) for x in re.findall(r'\d+', file)]
        return numbers

    image_files = sorted((os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".jpg")),
                         key=custom_sort)
    # while True:
    indices = list(range(0, len(image_files) - max(SEQUENCE_LENGTH, BATCH_SIZE)))
    # random.shuffle(indices)
    sequences = []
    labels = []
    skipped = []
    for st in tqdm(indices, leave=False, position=0):
        sequence = []
        for image_file in image_files[st:]:
            # print(image_file)
            image = get_image(image_file, IMAGE_HEIGHT, IMAGE_WIDTH)
            if is_almost_black(image):
                if image_file in skipped:
                    logger.info(f"{image_file} is skipped. Skipping")
                else:
                    skipped.append(image_file)
                    logger.info(f"{image_file} is almost entirely black. Skipping")
                continue
            elif len(sequence) > 0 and similarity(image, sequence[-1]) < 0.7:
                if not is_almost_black(sequence[-1]):
                    logger.info(f"{image_file}: too different with previous ({similarity(image, sequence[-1])})")
                break
            sequence.append(image)
            label = 1 if "sleep" in image_file else 0
            if len(sequence) == SEQUENCE_LENGTH:
                break
        if len(sequence) != SEQUENCE_LENGTH:
            logger.debug(f"Sequence not enough in length {len(sequence)}, skipping")
            continue
        sequences.append(sequence)
        labels.append(label)

        # yield np.array(sequence), label

    X, y = balance_classes(sequences, labels)
    # X = sequences
    # y = labels
    # for ix in range(0, 500, SEQUENCE_LENGTH * 10):
    #     display_image_grid_from_arrays(sequences_aug[ix:], rows=10)
    # for ix in range(0, 500, SEQUENCE_LENGTH * 10):
    #     display_image_grid_from_arrays(sequences[ix:], rows=10)
    print(f"There are {len(image_files)} images and {len(X)} sequences from which {sum(y)} are label 1 ({sum(y)/len(y)*100:.2f}%)")
    X = np.array(X, dtype=np.float16)
    y = np.array(y, dtype=np.float16)
    return X, y


SEED = np.random.randint(0, 1000000)
if __name__ == "__main__":
    pass
