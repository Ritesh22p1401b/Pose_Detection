import tensorflow as tf
import os
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================
# PATHS
# ==============================
BASE_DIR = "dataset"
TEST_DIR = os.path.join(BASE_DIR, "test")
CHECKPOINT_DIR = "checkpoints"

IMG_SIZE = 48
BATCH_SIZE = 64

# ==============================
# FIND LATEST CHECKPOINT
# ==============================
checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".keras")]
checkpoints.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])

print("[INFO] Evaluating:", latest_checkpoint)

# ==============================
# LOAD MODEL
# ==============================
model = tf.keras.models.load_model(latest_checkpoint)

# ==============================
# TEST DATA GENERATOR
# ==============================
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ==============================
# EVALUATE
# ==============================
test_loss, test_acc = model.evaluate(test_gen)

print(f"\n✅ Overall Test Accuracy: {test_acc * 100:.2f}%")
print(f"📉 Test Loss: {test_loss:.4f}")
