import os
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense,
    Dropout, Flatten, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# =====================================================
# GPU SETUP
# =====================================================
print("[INFO] TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("[INFO] GPU ENABLED")
else:
    print("[WARNING] GPU NOT FOUND")

# =====================================================
# PATHS
# =====================================================
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "validation")
TEST_DIR  = os.path.join(BASE_DIR, "test")

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =====================================================
# PARAMETERS
# =====================================================
IMG_SIZE = 48
BATCH_SIZE = 64
TOTAL_EPOCHS = 40
NUM_CLASSES = 7
LR = 0.0001

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

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =====================================================
# MODEL DEFINITION
# =====================================================
def build_model():
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
        optimizer=Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =====================================================
# FIND LAST CHECKPOINT
# =====================================================
def get_latest_checkpoint():
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".keras")]
    if not checkpoints:
        return None, 0

    checkpoints.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    latest = checkpoints[-1]
    epoch = int(re.findall(r"\d+", latest)[0])
    return os.path.join(CHECKPOINT_DIR, latest), epoch

checkpoint_path, last_epoch = get_latest_checkpoint()

if checkpoint_path:
    print(f"[INFO] Resuming from {checkpoint_path}")
    model = load_model(checkpoint_path)
    initial_epoch = last_epoch
else:
    print("[INFO] No checkpoint found. Starting fresh training.")
    model = build_model()
    initial_epoch = 0

model.summary()

# =====================================================
# CHECKPOINT CALLBACK (SAVE EVERY EPOCH)
# =====================================================
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(
        CHECKPOINT_DIR, "emotion_epoch_{epoch:02d}.keras"
    ),
    save_weights_only=False,   # IMPORTANT (saves optimizer state)
    save_freq="epoch",
    verbose=1
)

# =====================================================
# TRAIN (RESUMABLE)
# =====================================================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=TOTAL_EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=[checkpoint_callback]
)

# =====================================================
# SAVE FINAL MODEL
# =====================================================
FINAL_MODEL_PATH = "emotion_module/emotion_model_final.keras"
os.makedirs("emotion_module", exist_ok=True)
model.save(FINAL_MODEL_PATH)

print(f"[INFO] FINAL MODEL SAVED AT {FINAL_MODEL_PATH}")
