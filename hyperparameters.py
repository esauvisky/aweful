# These are linked to the dataset.
# You cannot change them without deleting
# and regenerating the entire dataset.
from keras.layers import ConvLSTM2D, Dense, Dropout, Flatten, Input, MaxPooling2D, TimeDistributed
from keras.models import Model
from keras.layers import Input, ConvLSTM2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam

SEQUENCE_LENGTH = 5
IMAGE_HEIGHT = 480 // 10
IMAGE_WIDTH = 640 // 10

# These are linked to the model architecture.
# You can change them experimentally, as long
# as the model does not change or you retrain it.
BATCH_SIZE = 8
EPOCHS = 1000
LEARNING_RATE = 0.001
THREADS = 32
DEBUG = False
DATASET_NAME = f'small-{IMAGE_HEIGHT}x{IMAGE_WIDTH}-{SEQUENCE_LENGTH}'
FILENAME = f'./prep/{DATASET_NAME}.h5'


def create_model():
    """Create a ConvLSTM model."""
    inputs = Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    x = ConvLSTM2D(filters=4, kernel_size=(3, 3), activation="tanh", return_sequences=True)(inputs)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(8, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    out_model = Model(inputs=inputs, outputs=outputs)
    out_model.summary()
    return out_model
