import pandas as pd
import joblib
import mlflow
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


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

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Pipeline
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

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)

    # Get classification report as dictionary
    report = classification_report(y_test, y_pred, output_dict=True)

    print("Accuracy: ", report["accuracy"])
    print("f1-score: ", report["weighted avg"]["f1-score"])

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", report["accuracy"])
    mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])
    print("\nMetrics logged to MLflow")

    # Save model
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train()
