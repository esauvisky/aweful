#!/usr/bin/env python3

import asyncio
import os
import re

import wandb
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


def display_single_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # Convert from BGR to RGB
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def display_image_grid(image_paths, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

    for idx, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # Convert from BGR to RGB
        ax = axes[idx // cols, idx % cols]
        ax.imshow(img)
        ax.axis("off")

    plt.show()



def display_image_grid_from_arrays(sequences, rows=10, cols=None):
    if cols is None:
        cols = SEQUENCE_LENGTH

    fig, axes = plt.subplots(rows, cols, figsize=(IMAGE_WIDTH, IMAGE_HEIGHT))

    for row in range(rows):
        for col in range(cols):
            if row < len(sequences) and col < len(sequences[row]):
                ax = axes[row, col]
                ax.imshow(sequences[row * SEQUENCE_LENGTH][col], cmap='gray')
                ax.axis("off")
            else:
                axes[row, col].remove()  # Remove unused subplots

    plt.waitforbuttonpress()
    plt.show()


def display_gif_grid_from_arrays(image_sequences, rows, cols, duration=100):
    gif_paths = []

    for idx, image_sequence in enumerate(image_sequences):
        gif_path = f"temp_gif_{idx}.gif"
        gif_paths.append(gif_path)

        images = [
            Image.fromarray((image_array.squeeze(-1)).astype(np.uint8), mode='L') for image_array in image_sequence]
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

    display_image_grid(gif_paths, rows, cols)

    for gif_path in gif_paths:
        os.remove(gif_path)
