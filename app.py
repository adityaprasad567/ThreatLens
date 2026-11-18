"""ThreatLens: URL phishing detection. Run: python -m streamlit run app.py"""
import json
from pathlib import Path
import streamlit as st
from src import database as db
from src.config import MODEL_PATH, METRICS_PATH, MODELS_DIR
from src.feature_extraction import FEATURE_NAMES
from src.prediction import load_model, predict_url

st.set_page_config(page_title='ThreatLens', page_icon='🛡️', layout='centered')
css = Path(__file__).parent / 'style.css'
if css.exists():
    st.markdown(f'<style>{css.read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)
st.title('🛡️ ThreatLens')
st.write('Machine learning phishing URL detection')
st.caption('URLs are analyzed as text. Websites are never opened.')

@st.cache_resource
def get_model():
    return load_model()

db.init_db()
if not MODEL_PATH.exists():
    st.error('Trained model missing. Run python -m src.train_model first.')
    st.stop()
try:
    model = get_model()
except Exception as error:
    st.error(f'Model loading failed: {error}')
    st.stop()

st.subheader('🔍 Scan a URL')
with st.form('scanner'):
    url = st.text_input('Website URL', placeholder='https://example.com')
    scan = st.form_submit_button('Check URL', type='primary', use_container_width=True)
if scan:
    try:
        result = predict_url(url.strip(), model)
        if not result.get('ok'):
            st.error(result.get('error', 'Invalid URL'))
        else:
            db.save_scan(result)
            st.subheader('📊 Prediction Result')
            if result['prediction'] == 'Phishing':
                st.error('⚠️ Potential phishing URL')
            else:
                st.success('✅ URL appears legitimate')
            a, b = st.columns(2)
            probability = float(result["phishing_probability"])
            raw_probability = float(result["model_probability"])

            st.write(f"Raw ML probability: {raw_probability:.1%}")
            st.write(f"Final phishing probability: {probability:.1%}")
            a.metric('Phishing probability', f'{probability:.1%}')
            b.metric('Risk level', result.get('risk_level', 'Unknown'))
            st.progress(max(0., min(1., probability)))
            for indicator in result.get('indicators', [])[:3]:
                if isinstance(indicator, dict):
                    st.markdown(f"**• {indicator.get('title', 'Indicator')}**")
                    st.caption(indicator.get('detail', ''))
                else:
                    st.write('•', str(indicator))
            st.info('Predictions are not safety guarantees. Verify links independently.')
    except Exception as error:
        st.error(f'Scan failed: {error}')

with st.expander('📈 Model Performance'):
    if METRICS_PATH.exists():
        try:
            data = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
            st.write('**Deployed model:**', data.get('model_name', 'Classifier'))
            scores = data.get('metrics', {})
            for col, (label, key) in zip(st.columns(4), [('Accuracy', 'accuracy'),
                    ('Precision', 'precision'), ('Recall', 'recall'), ('F1', 'f1')]):
                if key in scores:
                    col.metric(label, f'{scores[key]:.2%}')
        except (OSError, ValueError, TypeError) as error:
            st.warning(f'Metrics unavailable: {error}')
    else:
        st.info('No saved model metrics yet.')

with st.expander('🧪 Logistic Regression vs Random Forest'):
    comparison = MODELS_DIR / 'comparison_metrics.json'
    if comparison.exists():
        try:
            data = json.loads(comparison.read_text(encoding='utf-8'))
            st.caption(f"Separate experiment · {data['sample_rows']:,} URLs · {data['split']}")
            st.dataframe([{'Model': name, **{k.title(): f'{v:.2%}' for k, v in values.items()}}
                          for name, values in data['models'].items()], hide_index=True)
            st.caption('These models were compared separately; the deployed scanner uses its existing SGD model.')
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.warning(f'Comparison unavailable: {error}')
    else:
        st.info('Run python -m src.compare_models --data data/Combined_PhiUSIIL_PhiLegit.csv to generate a real comparison.')

with st.expander('🧩 URL Features'):
    st.write(f'{len(FEATURE_NAMES)} extracted URL features:')
    st.write(', '.join(FEATURE_NAMES))
    st.caption('HTTPS is extracted but excluded from model training to avoid misleading correlations.')

with st.expander('🕒 Recent Scan History'):
    try:
        history = db.get_history(limit=10)
        if history.empty:
            st.info('No scans yet.')
        else:
            st.dataframe(history[['scanned_at', 'url', 'prediction', 'risk_level']],
                         column_config={'scanned_at': 'Date & Time', 'url': 'Website URL',
                                        'prediction': 'Prediction', 'risk_level': 'Risk Level'},
                         hide_index=True, use_container_width=True)
    except Exception as error:
        st.warning(f'History unavailable: {error}')
st.divider()
st.caption('ThreatLens | Python · Scikit-learn · Streamlit · SQLite')
