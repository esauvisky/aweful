# These are linked to the dataset.
# You cannot change them without deleting
# and regenerating the entire dataset.
SEQUENCE_LENGTH = 8
IMAGE_HEIGHT = 480 // 2
IMAGE_WIDTH = 640 // 2

# These are linked to the model architecture.
# You can change them experimentally, as long
# as the model does not change or you retrain it.
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 20
THREADS = 32
DEBUG = False
FILENAME = f'./prep/weights-{SEQUENCE_LENGTH}_{IMAGE_HEIGHT}_{IMAGE_WIDTH}.h5'
