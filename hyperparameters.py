# These are linked to the dataset.
# You cannot change them without deleting
# and regenerating the entire dataset.
SEQUENCE_LENGTH = 8
IMAGE_HEIGHT = 480 // 10
IMAGE_WIDTH = 640 // 10

# These are linked to the model architecture.
# You can change them experimentally, as long
# as the model does not change or you retrain it.
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-1
THREADS = 16
DEBUG = False
DATASET_NAME = f'small-{IMAGE_HEIGHT}x{IMAGE_WIDTH}'
FILENAME = f'./prep/{DATASET_NAME}-{SEQUENCE_LENGTH}.h5'
