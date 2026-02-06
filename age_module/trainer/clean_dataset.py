import os
import cv2
import pandas as pd
from tqdm import tqdm

# ================= CONFIG =================
DATASETS_DIR = "datasets"

IMAGES_ROOT = os.path.join(DATASETS_DIR, "imdb-clean-1024")

TRAIN_CSV_IN = os.path.join(DATASETS_DIR, "imdb_train_new_1024.csv")
VAL_CSV_IN   = os.path.join(DATASETS_DIR, "imdb_valid_new_1024.csv")

TRAIN_CSV_OUT = os.path.join(DATASETS_DIR, "imdb_train_cleaned_1024.csv")
VAL_CSV_OUT   = os.path.join(DATASETS_DIR, "imdb_valid_cleaned_1024.csv")

MIN_AGE = 0
MAX_AGE = 80
MIN_IMG_SIZE = 40
# =========================================


def find_image_path(img_name: str):
    """
    Tries to locate image inside 00–99 subfolders.
    """
    # If CSV already contains subfolder (e.g. 12/xxx.jpg)
    direct_path = os.path.join(IMAGES_ROOT, img_name)
    if os.path.exists(direct_path):
        return direct_path, img_name

    # Otherwise search 00–99
    for i in range(100):
        folder = f"{i:02d}"
        path = os.path.join(IMAGES_ROOT, folder, img_name)
        if os.path.exists(path):
            return path, f"{folder}/{img_name}"

    return None, None


def is_valid_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    h, w, _ = img.shape
    return h >= MIN_IMG_SIZE and w >= MIN_IMG_SIZE


def clean_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Detect image column
    if "path" in df.columns:
        img_col = "path"
    elif "image" in df.columns:
        img_col = "image"
    else:
        raise ValueError(
            f"No image column found in {input_csv}. "
            f"Columns: {list(df.columns)}"
        )

    if "age" not in df.columns:
        raise ValueError("CSV must contain an 'age' column")

    cleaned = []

    print(f"\n[INFO] Cleaning {os.path.basename(input_csv)}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            age = int(row["age"])
            if age < MIN_AGE or age > MAX_AGE:
                continue

            img_name = str(row[img_col]).strip()
            img_path, rel_path = find_image_path(img_name)

            if img_path is None:
                continue

            if not is_valid_image(img_path):
                continue

            cleaned.append({
                "path": rel_path,
                "age": age
            })

        except Exception:
            continue

    if len(cleaned) == 0:
        raise RuntimeError(
            f"❌ Cleaning removed ALL samples from {input_csv}. "
            f"Check image paths or folder structure."
        )

    out_df = pd.DataFrame(cleaned)
    out_df.to_csv(output_csv, index=False)

    print(f"[DONE] Saved {output_csv}")
    print(f"[STATS] Kept {len(out_df)} / {len(df)} samples")


def main():
    clean_csv(TRAIN_CSV_IN, TRAIN_CSV_OUT)
    clean_csv(VAL_CSV_IN, VAL_CSV_OUT)
    print("\n✅ Dataset cleaning completed successfully")


if __name__ == "__main__":
    main()
