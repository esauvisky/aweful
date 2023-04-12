import cv2
from keras.preprocessing.image import ImageDataGenerator, image_utils
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np

from tqdm import tqdm


def get_image(image_path, image_height, image_width):
    image = image_utils.load_img(
        image_path,
        color_mode="grayscale",
    )
    image = image[0:image.shape[0], 0:image.shape[1] - 21]
    image_utils.smart_resize(image, (image_height, image_width))
    image = image_utils.img_to_array(image) / 255.0
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (image_width, image_height))
    image = np.expand_dims(image, axis=-1) # Add an extra channel dimension
    return image


def process_image_sequence(image_files, images_path, image_height, image_width):
    images = []
    labels = []

    for file in image_files:
        image = get_image(os.path.join(images_path, file), image_height, image_width)
        label = 1 if "sleep" in file else 0
        images.append(image)
        labels.append(label)

    return images, labels[-1]


def load_data(images_path, seq_length, image_height, image_width):
    X, y = [], []

    # Get the list of image files
    image_files = sorted((file for file in os.listdir(images_path) if file.endswith(".jpg")),
                         key=lambda x: int(x.split(".")[0].split("-")[0]))

    # Use a ThreadPoolExecutor to process image sequences in parallel
    with ThreadPoolExecutor(max_workers=16) as executor:
        # Create the tqdm progress bar
        progress_bar = tqdm(
            total=len(image_files) - seq_length,
            smoothing=0.1,
            desc="Processing images...",
        )

        # Submit the tasks to the executor
        futures = [
            executor.submit(process_image_sequence, image_files[i:i + seq_length], images_path, image_height, image_width)
            for i in range(0,
                           len(image_files) - seq_length)]

        # Use as_completed() to process the results as they become available, and update the progress bar
        for future in as_completed(futures):
            images, label = future.result()
            X.append(images)
            y.append(label)
            progress_bar.update(1)

        progress_bar.close()

    X = np.array(X)
    y = np.array(y)
    return X, y
