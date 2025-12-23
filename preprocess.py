import pandas as pd
import re
import argparse
import zipfile
from pathlib import Path


# -------------------------
# Default paths
# -------------------------
DEFAULT_ZIP_PATH = "/mnt/input/sentiment140"
DEFAULT_EXTRACT_DIR = "/tmp/sentiment140"
DEFAULT_OUTPUT_PATH = "/mnt/output/clean_sentiment.csv"

CSV_NAME = "training.1600000.processed.noemoticon.csv"


# -------------------------
# Text cleaning
# -------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)   # URLs
    text = re.sub(r"@\w+", "", text)             # mentions
    text = re.sub(r"#\w+", "", text)             # hashtags
    text = re.sub(r"[^a-z\s]", "", text)         # non-alpha
    text = re.sub(r"\s+", " ", text).strip()     # whitespace

    return text


def extract_zip(zip_path: Path, extract_dir: Path):
    if extract_dir.exists():
        print(f"ZIP already extracted at: {extract_dir}")
        return

    print(f"Extracting ZIP: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    print("Extraction completed.")


def preprocess(zip_path: str, output_path: str):
    zip_path = Path(zip_path)
    extract_dir = Path(DEFAULT_EXTRACT_DIR)
    output_path = Path(output_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    # Extract ZIP if needed
    extract_zip(zip_path, extract_dir)

    csv_path = extract_dir / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}")

    print(f"Loading CSV: {csv_path}")

    df = pd.read_csv(
        csv_path,
        encoding="latin-1",
        header=None
    )

    df.columns = ["sentiment", "id", "date", "query", "user", "text"]

    # Keep valid labels
    df = df[df["sentiment"].isin([0, 4])]

    # Convert labels: 0 → negative, 4 → positive
    df["sentiment"] = df["sentiment"].map({0: 0, 4: 1})

    # Keep only required columns
    df = df[["sentiment", "text"]]

    print("Cleaning text...")
    df["text"] = df["text"].apply(clean_text)

    # Drop empty rows
    df = df[df["text"].str.len() > 0]

    print(f"Final row count: {len(df)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Cleaned dataset saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Sentiment140 ZIP dataset")
    parser.add_argument(
        "--input",
        default=DEFAULT_ZIP_PATH,
        help="Path to sentiment140.zip"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output cleaned CSV path"
    )

    args = parser.parse_args()
    preprocess(args.input, args.output)

