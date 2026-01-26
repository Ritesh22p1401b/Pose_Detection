import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense,
    Dropout, Flatten, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# =====================================================
# GPU / CPU CONFIGURATION (AUTO)
# =====================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU detected. CUDA enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("[INFO] No GPU detected. Using CPU.")

# =====================================================
# PATHS
# =====================================================
BASE_DIR = "datasets/emotion"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# =====================================================
# PARAMETERS
# =====================================================
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 40
NUM_CLASSES = 7
LEARNING_RATE = 0.0001

# =====================================================
# DATA GENERATORS
# =====================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =====================================================
# CNN MODEL
# =====================================================
model = Sequential([
    Conv2D(64, (3,3), activation="relu", input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),

    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# TRAINING
# =====================================================
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS
)

# =====================================================
# SAVE MODEL
# =====================================================
MODEL_PATH = "emotion/emotion_model.h5"
model.save(MODEL_PATH)
print(f"[INFO] Emotion model saved at: {MODEL_PATH}")
