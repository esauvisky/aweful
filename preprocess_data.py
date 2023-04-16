#!/usr/bin/env python3

import os

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

from hyperparameters import SEQUENCE_LENGTH, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, THREADS, EPOCHS, LEARNING_RATE, PATIENCE, DEBUG, FILENAME

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
    datagen = ImageDataGenerator(height_shift_range=(-0.18, 0.10),
                                 width_shift_range=0.1,
                                 zoom_range=0.05,
                                 fill_mode='reflect')

    # Add an extra dimension for the batch size
    images = np.array(images)

    transformation_matrix = datagen.get_random_transform(images.shape[1:])

    augmented_images = []
    for image in images:
        augmented_image = datagen.apply_transform(image, transformation_matrix)
        # show_image(augmented_image)
        augmented_images.append(augmented_image.astype(np.int32))

    return augmented_images


def show_image(image):
    # Read the input image
    # image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if image.dtype != np.int32:
        image = (image * 255).astype(np.int32)

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
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

    image = np.expand_dims(image, axis=-1)
    return image


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0)**invGamma) * 255 for i in np.arange(0, 256)])
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def get_image(image_path, image_height, image_width):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image[0:image.shape[0] - 21, 0:image.shape[1]]
    image = cv2.resize(image, (image_width, image_height))
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel1)
    # image = image / close
    # image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
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

    augmented_sequence = simulate_panning(sequence)

    return sequence, augmented_sequence, labels[-1]


def process_data(input_dir):
    global SEED
    X, X_aug, y = [], [], []

    # Get the list of image files
    image_files = sorted((file for file in os.listdir(input_dir) if file.endswith(".jpg")),
                         key=lambda x: int(x.split(".")[0].split("-")[0]))

    minority_image_files = [file for file in image_files if "sleep" in file]
    majority_image_files = [file for file in image_files if "sleep" not in file]

    majority_class_count = len(majority_image_files)
    minority_class_count = len(minority_image_files)
    oversampling_factor = majority_class_count // minority_class_count

    sequences_filenames = []
    minority_count = 0
    majority_count = 0
    for i in range(len(image_files) - SEQUENCE_LENGTH):
        sequence = image_files[i:i + SEQUENCE_LENGTH]
        if "sleep" in sequence[-1]:
            for _ in range(oversampling_factor // 3):
                minority_count += 1
                sequences_filenames.append(sequence)
        elif random.random() < 0.5:
            majority_count += 1
            sequences_filenames.append(sequence)

    # remove the final sequences until we have a multiple of the batch size
    sequences_filenames = sequences_filenames[:len(sequences_filenames) - (len(sequences_filenames) % BATCH_SIZE)]

    logger.info(f"Oversampled sequences: {len(sequences_filenames)}")
    logger.info(f"Minority count: {minority_count}")
    logger.info(f"Majority count: {majority_count}")

    # Use a ThreadPoolExecutor to process image sequences in parallel
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        # Create the tqdm progress bar
        progress_bar = tqdm(total=len(sequences_filenames) - SEQUENCE_LENGTH,
                            smoothing=0.1,
                            desc="Processing images...",
                            position=0,
                            leave=True)

        offset = np.random.randint(0, 200)
        futures = [
            executor.submit(process_image_sequence, sequences_filenames[i], input_dir, IMAGE_HEIGHT, IMAGE_WIDTH)
            for i in range(0,
                           len(sequences_filenames) - SEQUENCE_LENGTH)]

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

    # logger.info(f"X_aug shape: {X_aug.shape}")
    # logger.info("Concatenating...")
    # X = np.concatenate((X, X_aug))
    # y = np.concatenate((y, y))
    logger.info(f"X_aug size: {len(X_aug)} | X Actual: {len(X)} | Overlook sequences: {len(sequences_filenames) - SEQUENCE_LENGTH}")
    logger.info(f"y size: {len(y)} | Classes: {np.unique(y)} | Counts: {np.bincount(y)}")

    return X, X_aug, y


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
    labels = np.array(labels, dtype=np.int32)
    return sequences, labels


def load_individual_file(input_dir, idx):
    sequence_file = os.path.join(input_dir, f"sequence_{idx}.npz")
    sequence = (np.load(sequence_file)["sequence"]).astype(np.float32)

    label_file = os.path.join(input_dir, f"label_{idx}.npy")
    label = np.load(label_file)

    return sequence, label


def load_sequences(key, datatype):
    input_dir = os.path.join("./prep/", key)
    idxs = get_generator_idxs(key, datatype)

    for idx in idxs:
        yield load_individual_file(input_dir, idx)


# Returns a list of sequence indices
def get_generator_idxs(key, datatype):
    input_dir = os.path.join("./prep/", key)
    num_files = len(os.listdir(input_dir)) // 2

    if datatype == "val":
        return range(0, int(num_files * 0.25))
    elif datatype == "wandb":
        return np.random.permutation(range(0, num_files))[:BATCH_SIZE * 3]
    elif datatype == "train":
        return np.random.permutation(range(int(num_files * 0.25), num_files))
    else:
        raise ValueError("Incorrect datatype for generator: {}".format(datatype))


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
    sequences, augmented_sequences, labels = process_data(f"./data/{key}")
    sequences = sequences
    augmented_sequences = augmented_sequences

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
        table.add_data(labels[ix], get_video(augmented_sequences[ix]))

    wandb.log({key: table})

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        save_futures = [
            executor.submit(save_single_data, sequences[i], labels[i], i, key) for i in range(len(sequences))]
        save_futures += [
            executor.submit(save_single_data, augmented_sequences[i], labels[i]) for i in range(len(augmented_sequences))]

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
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"Num GPUs Available: {len(gpus)}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    save_data("raw")
    wandb.finish()
