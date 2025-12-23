import joblib
import argparse


# -------------------------
# Default model path
# -------------------------
DEFAULT_MODEL_PATH = "/mnt/models/model.joblib"


def predict(texts, model_path):
    model = joblib.load(model_path)

    # Predict label and probability
    preds = model.predict(texts)
    probs = model.predict_proba(texts)

    results = []
    for text, label, prob in zip(texts, preds, probs):
        sentiment = "POSITIVE" if label == 1 else "NEGATIVE"
        confidence = prob[1]  # probability of positive class

        results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4)
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment prediction")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to trained .joblib model"
    )
    parser.add_argument(
        "--text",
        nargs="+",
        required=True,
        help="Text(s) to classify"
    )

    args = parser.parse_args()

    outputs = predict(args.text, args.model)

    for out in outputs:
        print(
            f"[{out['sentiment']}] "
            f"(confidence={out['confidence']}) → {out['text']}"
        )

#python predict.py --model /mnt/models/model.joblib  --text   "this product is terrible"   "i absolutely love this phone"
