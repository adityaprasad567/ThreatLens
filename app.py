
"""ThreatLens - Machine Learning Phishing URL Detector.
Run: python -m streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

from src.config import MODEL_PATH, METRICS_PATH
from src.prediction import load_model, predict_url
from src import database as db


# ---------- PAGE SETUP ----------
st.set_page_config(
    page_title="ThreatLens",
    page_icon="🛡️",
    layout="centered"
)

# Load custom CSS
css_file = Path(__file__).parent / "style.css"

if css_file.exists():
    st.markdown(
        f"<style>{css_file.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True
    )

st.title("🛡️ ThreatLens")
st.write("AI-powered phishing URL detection using machine learning.")
st.caption("Analyze suspicious links without opening the website.")


# ---------- LOAD MODEL ----------
@st.cache_resource
def get_model():
    return load_model()


db.init_db()

if not MODEL_PATH.exists():
    st.error("Model not found. Run: python -m src.train_model")
    st.stop()

try:
    model = get_model()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# ---------- URL SCANNER ----------
st.subheader("🔍 Scan a URL")

with st.form("scanner"):
    url = st.text_input(
        "Enter a website URL",
        placeholder="https://example.com"
    )
    scan = st.form_submit_button(
        "🔍 Check URL",
        type="primary",
        use_container_width=True
    )

if scan:
    try:
        result = predict_url(url.strip(), model)

        if not result.get("ok"):
            st.error(result.get("error", "Invalid URL"))

        else:
            db.save_scan(result)

            prediction = result["prediction"]
            probability = float(result["phishing_probability"])
            risk = result.get("risk_level", "Unknown")

            st.subheader("📊 Prediction Result")

            if prediction == "Phishing":
                st.error("⚠️ Potential phishing URL detected")
            else:
                st.success("✅ URL appears legitimate")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Phishing Probability",
                    f"{probability:.1%}"
                )

            with col2:
                st.metric("Risk Level", risk)

            st.progress(max(0.0, min(1.0, probability)))

            # Show detection indicators clearly
            indicators = result.get("indicators", [])

            if indicators:
                st.markdown("**🔎 Detection Indicators**")

                for indicator in indicators[:3]:
                    if isinstance(indicator, dict):
                        title = indicator.get(
                            "title", "Indicator"
                        )
                        detail = indicator.get(
                            "detail", ""
                        )

                        st.markdown(f"🔹 **{title}**")

                        if detail:
                            st.caption(detail)
                    else:
                        st.write("🔹", str(indicator))

            st.info(
                "A prediction is not a safety guarantee. "
                "Always verify suspicious links."
            )

    except Exception as error:
        st.error(f"Scan failed: {error}")


# ---------- MODEL PERFORMANCE ----------
with st.expander("📈 Model Performance"):
    if METRICS_PATH.exists():
        try:
            data = json.loads(
                METRICS_PATH.read_text(encoding="utf-8")
            )

            scores = data.get("metrics", {})

            st.write(
                "**Model:**",
                data.get("model_name", "Trained classifier")
            )

            metrics = [
                ("Accuracy", "accuracy"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1 Score", "f1"),
            ]

            cols = st.columns(4)

            for col, (label, key) in zip(cols, metrics):
                if key in scores:
                    col.metric(
                        label,
                        f"{scores[key]:.2%}"
                    )

        except (OSError, ValueError, TypeError) as error:
            st.warning(f"Metrics unavailable: {error}")
    else:
        st.info("Train the model to generate metrics.")


# ---------- SCAN HISTORY ----------
with st.expander("🕒 Recent Scan History"):
    try:
        history = db.get_history(limit=10)

        if history.empty:
            st.info("No scans yet. Scan a URL to begin.")

        else:
            columns = [
                "scanned_at",
                "url",
                "prediction",
                "risk_level"
            ]

            st.dataframe(
                history[columns],
                column_config={
                    "scanned_at": "Date & Time",
                    "url": "Website URL",
                    "prediction": "Prediction",
                    "risk_level": "Risk Level"
                },
                hide_index=True,
                use_container_width=True
            )

    except Exception as error:
        st.warning(f"Could not load scan history: {error}")


# ---------- FOOTER ----------
st.divider()

st.caption(
    "🛡️ ThreatLens | Built with Python, Scikit-learn, "
    "Streamlit and SQLite"
)
