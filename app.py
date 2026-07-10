import joblib
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

MODEL_PATH = Path("models") / "sms_spam_model.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model not found. Please train the model first using: python src/train.py")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("📩 SMS Spam Detection App")
st.write("Enter an SMS message below and click Predict.")

message = st.text_area(
    "SMS Message",
    height=150,
    placeholder="Example: Congratulations! You have won a free gift voucher. Click now!"
)

threshold = st.slider(
    "Spam Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.35,
    step=0.05,
    help="Lower threshold catches more spam, but may also classify some normal messages as spam."
)

if st.button("Predict"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        proba = model.predict_proba([message])[0]
        spam_prob = float(proba[1])

        if spam_prob >= threshold:
            label = "SPAM 🚨"
        else:
            label = "HAM ✅"

        st.subheader("Prediction Result")
        st.success(f"Prediction: {label}")
        st.write(f"Spam Probability: **{spam_prob:.3f}**")
        st.write(f"Threshold Used: **{threshold:.2f}**")

        if label.startswith("SPAM"):
            st.info("This message appears suspicious based on the trained model.")
        else:
            st.info("This message appears to be a normal SMS.")
