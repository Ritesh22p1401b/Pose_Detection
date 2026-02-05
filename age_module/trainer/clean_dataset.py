import scipy.io
import os
import cv2
import pandas as pd
from datetime import datetime

DATASET_ROOT = "datasets/imdb_wiki"
MAT_FILE = os.path.join(DATASET_ROOT, "imdb.mat")
IMAGE_ROOT = os.path.join(DATASET_ROOT, "imdb_crop")

def calculate_age(dob, photo_year):
    birth_year = datetime.fromordinal(int(dob)).year
    return photo_year - birth_year

def is_valid_face(img):
    if img is None:
        return False
    h, w, _ = img.shape
    return h >= 40 and w >= 40

def main():
    mat = scipy.io.loadmat(MAT_FILE)
    imdb = mat["imdb"][0][0]

    dob = imdb[0][0]
    photo_year = imdb[1][0]
    paths = imdb[2][0]

    records = []

    for i in range(len(paths)):
        try:
            age = calculate_age(dob[i], photo_year[i])
            if age < 0 or age > 100:
                continue

            img_path = os.path.join(IMAGE_ROOT, paths[i][0])
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if not is_valid_face(img):
                continue

            records.append((img_path, age))
        except:
            continue

    df = pd.DataFrame(records, columns=["path", "age"])
    out_csv = os.path.join(DATASET_ROOT, "clean_imdb.csv")
    df.to_csv(out_csv, index=False)

    print(f"[INFO] Clean dataset saved: {len(df)} samples")

if __name__ == "__main__":
    main()
