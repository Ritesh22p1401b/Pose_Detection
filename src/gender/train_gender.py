import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# -------- CONFIG --------
CSV_PATH = "datasets/gender_classification.csv"
MODEL_DIR = "checkpoints"
MODEL_PATH = os.path.join(MODEL_DIR, "gender_model.pkl")

LABEL_COLUMN = "gender"   # change if your column name differs
# ------------------------


def main():
    print("Loading gender classification dataset...")
    df = pd.read_csv(CSV_PATH)

    print("Dataset shape:", df.shape)

    # -------- LABEL HANDLING --------
    if df[LABEL_COLUMN].dtype == object:
        df[LABEL_COLUMN] = df[LABEL_COLUMN].str.lower().map(
            {"male": 0, "female": 1}
        )

    X = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN]

    print("Features:", X.shape[1])

    # -------- SPLIT --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------- MODEL PIPELINE --------
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Training gender classifier...")
    pipeline.fit(X_train, y_train)

    # -------- EVALUATION --------
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nGender Classification Accuracy:", round(acc * 100, 2), "%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Male", "Female"]))

    # -------- SAVE MODEL --------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print("\nGender model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
