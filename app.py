"""ThreatLens: simple phishing URL detection dashboard.
Run: streamlit run app.py
"""

import json
import streamlit as st
from src.config import MODEL_PATH, METRICS_PATH
from src.prediction import load_model, predict_url
from src import database as db

st.set_page_config(page_title="ThreatLens", page_icon="🛡️", layout="centered")
st.title("🛡️ ThreatLens")
st.write("Check whether a URL looks like a phishing link using machine learning.")
st.caption("The URL is analyzed as text; the website is not opened.")

# Load the trained model once, instead of on every button click.
@st.cache_resource

def get_model():
    return load_model()

# Initialize the local database used to save scan results.
db.init_db()

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run: python -m src.train_model")
    st.stop()

try:
    model = get_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

# Main scanner
st.subheader("Scan a URL")
with st.form("scanner"):
    url = st.text_input("Enter a website URL", placeholder="https://example.com")
    scan = st.form_submit_button("Check URL", type="primary")

if scan:
    try:
        result = predict_url(url.strip(), model)
        if not result.get("ok"):
            st.error(result.get("error", "Invalid URL"))
        else:
            db.save_scan(result)
            prediction = result["prediction"]
            probability = float(result["phishing_probability"])
            st.subheader("Prediction result")
            if prediction == "Phishing":
                st.error("⚠️ Potential phishing URL")
            else:
                st.success("✅ URL appears legitimate")
            st.metric("Phishing probability", f"{probability:.1%}")
            st.progress(max(0.0, min(1.0, probability)))
            st.write("Risk level:", result.get("risk_level", "Unknown"))
            for indicator in result.get("indicators", [])[:3]:
                st.write("•", str(indicator))
            st.info("A prediction is not a safety guarantee. Verify links independently.")
    except Exception as error:
        st.error(f"Scan failed: {error}")

# Model evaluation numbers from the existing training output.
with st.expander("Model performance"):
    if METRICS_PATH.exists():
        try:
            data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            scores = data.get("metrics", {})
            st.write("Model:", data.get("model_name", "Trained classifier"))
            for label, key in [
                ("Accuracy", "accuracy"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1 score", "f1"),
            ]:
                if key in scores:
                    st.metric(label, f"{scores[key]:.2%}")
        except (OSError, ValueError, TypeError) as error:
            st.warning(f"Metrics unavailable: {error}")
    else:
        st.info("Train the model to generate performance metrics.")

# Show recent scans saved in SQLite.
with st.expander("Recent scan history"):
    try:
        history = db.get_history(limit=10)
        if history.empty:
            st.info("No scans yet. Scan a URL to begin.")
        else:
            columns = ["scanned_at", "url", "prediction", "risk_level"]
            st.dataframe(history[columns], hide_index=True, use_container_width=True)
    except Exception as error:
        st.warning(f"Could not read scan history: {error}")
