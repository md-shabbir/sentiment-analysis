import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# -------------------------
# Default paths
# -------------------------
CLEAN_DATA_PATH = "/mnt/input/clean_sentiment.csv"
MODEL_PATH = "/mnt/models/model.joblib"


def train():
    print(f"Loading cleaned data from: {CLEAN_DATA_PATH}")

    df = pd.read_csv(CLEAN_DATA_PATH)

    X = df["text"]
    y = df["sentiment"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            min_df=5
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    print("Training model on full dataset...")
    pipeline.fit(X, y)

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train()
