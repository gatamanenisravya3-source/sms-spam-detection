import joblib
from pathlib import Path

MODEL_PATH = Path("models") / "sms_spam_model.joblib"

def predict_message(msg: str):
    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba([msg])[0]
    pred = model.predict([msg])[0]
    threshold = 0.35
    label = "SPAM" if proba[1] >= threshold else "HAM"
    return label, float(proba[1])  # probability of spam

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Train first: python src/train.py")

    print("Type an SMS message (or press Enter to quit):\n")
    while True:
        msg = input("> ").strip()
        if not msg:
            break
        label, spam_prob = predict_message(msg)
        print(f"Prediction: {label} (spam probability: {spam_prob:.3f})\n")
