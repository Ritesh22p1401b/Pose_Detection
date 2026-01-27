import os
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# =====================================================
# PATHS (DATASETS OUTSIDE emotion_module)
# =====================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "emotion")
TEST_DIR = os.path.join(DATASET_DIR, "test")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

IMG_SIZE = 48
BATCH_SIZE = 64

# =====================================================
# FIND LATEST CHECKPOINT
# =====================================================
if not os.path.exists(CHECKPOINT_DIR):
    raise FileNotFoundError(f"Checkpoint directory not found: {CHECKPOINT_DIR}")

checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".keras")]

if not checkpoints:
    raise FileNotFoundError("No .keras model found in checkpoints folder")

checkpoints.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])

print(f"[INFO] Using model: {latest_checkpoint}")

# =====================================================
# LOAD MODEL
# =====================================================
model = tf.keras.models.load_model(latest_checkpoint)
print("[INFO] Model loaded successfully")

# =====================================================
# TEST DATA GENERATOR
# =====================================================
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_labels = list(test_gen.class_indices.keys())
print("[INFO] Class indices:", test_gen.class_indices)

# =====================================================
# EVALUATION
# =====================================================
test_loss, test_acc = model.evaluate(test_gen, verbose=1)

print("\n==============================")
print(f"✅ Test Accuracy : {test_acc * 100:.2f}%")
print(f"📉 Test Loss     : {test_loss:.4f}")
print("==============================\n")

# =====================================================
# PREDICTIONS
# =====================================================
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================
print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# =====================================================
# CONFUSION MATRIX
# =====================================================
cm = confusion_matrix(y_true, y_pred)
print("🧮 Confusion Matrix:")
print(cm)
