import os

# --------------------------------------------------
# PUT YOUR REAL CLASS NAMES HERE
# (NOT EPOCHS)
# --------------------------------------------------
LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
    # add ALL classes you trained on
]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EMOTION_MODULE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CHECKPOINT_DIR = os.path.join(EMOTION_MODULE_DIR, "checkpoints")
LABELS_PATH = os.path.join(CHECKPOINT_DIR, "labels.txt")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    for label in LABELS:
        f.write(label + "\n")

print("✅ labels.txt created")
print("📍", LABELS_PATH)
print("🔢 Total labels:", len(LABELS))
