import os
import cv2
import pandas as pd
from tqdm import tqdm

# ================= CONFIG =================
DATASETS_DIR = "datasets"

IMAGES_ROOT = os.path.join(
    DATASETS_DIR, "imdb-clean-1024", "imdb-clean-1024"
)

TRAIN_CSV_IN = os.path.join(DATASETS_DIR, "imdb_train_new_1024.csv")
VAL_CSV_IN   = os.path.join(DATASETS_DIR, "imdb_valid_new_1024.csv")

TRAIN_CSV_OUT = os.path.join(DATASETS_DIR, "imdb_train_cleaned_1024.csv")
VAL_CSV_OUT   = os.path.join(DATASETS_DIR, "imdb_valid_cleaned_1024.csv")

MIN_AGE = 0
MAX_AGE = 80
MIN_FACE_SIZE = 40
# =========================================


def normalize_filename(name):
    """
    Extract clean basename from CSV entry.
    Handles paths like:
    - 00/xxx.jpg
    - imdb-clean-1024/00/xxx.jpg
    """
    return os.path.basename(str(name)).strip().lower()


def build_image_index():
    """
    Build basename -> relative path index (robust).
    """
    print("[INFO] Building image index...")
    print("[DEBUG] IMAGES_ROOT =", os.path.abspath(IMAGES_ROOT))

    if not os.path.isdir(IMAGES_ROOT):
        raise RuntimeError(f"❌ IMAGES_ROOT not found: {IMAGES_ROOT}")

    index = {}

    for root, _, files in os.walk(IMAGES_ROOT):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, IMAGES_ROOT)
                index[fname.lower()] = rel_path

    print(f"[INFO] Indexed {len(index)} images")

    if len(index) == 0:
        raise RuntimeError("❌ Image index is empty. Dataset not detected.")

    return index


def is_valid_face_bbox(row):
    try:
        w = int(row["x_max"]) - int(row["x_min"])
        h = int(row["y_max"]) - int(row["y_min"])
        return w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE
    except Exception:
        return False


def is_valid_image(path):
    img = cv2.imread(path)
    return img is not None


def clean_csv(input_csv, output_csv, image_index):
    df = pd.read_csv(input_csv)

    required = {"filename", "age", "x_min", "y_min", "x_max", "y_max"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    cleaned = []

    print(f"\n[INFO] Cleaning {os.path.basename(input_csv)}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            age = int(row["age"])
            if age < MIN_AGE or age > MAX_AGE:
                continue

            if not is_valid_face_bbox(row):
                continue

            fname = normalize_filename(row["filename"])
            rel_path = image_index.get(fname)
            if rel_path is None:
                continue

            abs_path = os.path.join(IMAGES_ROOT, rel_path)
            if not is_valid_image(abs_path):
                continue

            cleaned.append({
                "path": rel_path.replace("\\", "/"),
                "age": age
            })

        except Exception:
            continue

    if not cleaned:
        raise RuntimeError(
            f"❌ No images resolved from {input_csv}\n"
            f"Image index size = {len(image_index)}\n"
            f"➡ CSV filenames do not match disk images"
        )

    out_df = pd.DataFrame(cleaned)
    out_df.to_csv(output_csv, index=False)

    print(f"[DONE] Saved {output_csv}")
    print(f"[STATS] Kept {len(out_df)} / {len(df)} samples")


def main():
    image_index = build_image_index()
    clean_csv(TRAIN_CSV_IN, TRAIN_CSV_OUT, image_index)
    clean_csv(VAL_CSV_IN, VAL_CSV_OUT, image_index)
    print("\n✅ Dataset cleaning completed successfully")


if __name__ == "__main__":
    main()
