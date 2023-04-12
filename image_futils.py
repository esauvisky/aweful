def get_random_crop(image, seed=None):
    if seed:
        np.random.seed(seed)

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
    resized_crop = np.expand_dims(resized_crop, axis=-1) # Add an extra channel dimension
    return resized_crop
