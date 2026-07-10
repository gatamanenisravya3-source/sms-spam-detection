import os
import re
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Step A: Download dataset
# -----------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_PATH = DATA_DIR / "sms_spam_collection.zip"
TXT_PATH = DATA_DIR / "SMSSpamCollection"

def download_and_extract():
    if TXT_PATH.exists():
        print(f"[OK] Dataset already exists: {TXT_PATH}")
        return

    print("[INFO] Downloading dataset...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print(f"[OK] Downloaded: {ZIP_PATH}")

    print("[INFO] Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    print("[OK] Extracted to data/")

# -----------------------------
# Step B: Load + clean
# -----------------------------
def basic_clean(text: str) -> str:
    # light cleaning only; TF-IDF handles a lot
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)     # normalize links
    text = re.sub(r"\d+", " NUM ", text)                 # normalize numbers
    text = re.sub(r"[^a-z\s]", " ", text)                # keep letters/spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data():
    df = pd.read_csv(TXT_PATH, sep="\t", header=None, names=["label", "message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["message"] = df["message"].astype(str).apply(basic_clean)
    return df

# -----------------------------
# Step C: Train model pipeline
# -----------------------------
def train():
    download_and_extract()
    df = load_data()

    X = df["message"]
    y = df["label"]

    # stratify keeps spam/ham ratio similar in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),      # use unigrams + bigrams
            min_df=2,                # ignore extremely rare tokens
            max_df=0.95              # ignore overly common tokens
        )),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))

    ])

    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    print("[INFO] Evaluating...")
    preds = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

    # Save model
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "sms_spam_model.joblib"
    joblib.dump(model, model_path)
    print(f"\n[OK] Saved model to: {model_path}")

if __name__ == "__main__":
    train()
