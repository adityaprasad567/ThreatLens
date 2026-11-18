"""ThreatLens - AI-powered phishing URL detection dashboard.

Run with:  streamlit run app.py
"""

import json

import pandas as pd
import streamlit as st
from PIL import Image

from src import charts, database as db
from src.config import (
    ASSETS_DIR, DEMO_URLS, EXAMPLE_URLS, METRICS_PATH, MODEL_PATH,
)
from src.feature_extraction import FEATURE_LABELS, FEATURE_NAMES
from src.prediction import ModelNotFoundError, load_model, predict_url
from src.train_model import pick_dataset, train
from src.ui import (
    callout, chip, html, indicator_rows, inject_css, kpi, page_header,
    sample_data_warning, section_title, url_anatomy, verdict_card, esc, domain_mode_note,
)

# ------------------------------------------------------------------ setup
try:
    _icon = Image.open(ASSETS_DIR / "logo.png")
except OSError:
    _icon = "🛡️"
st.set_page_config(page_title="ThreatLens", page_icon=_icon, layout="wide", initial_sidebar_state="auto")
inject_css()
db.init_db()
if (ASSETS_DIR / "logo_wide.png").exists():
    st.logo(str(ASSETS_DIR / "logo_wide.png"), icon_image=str(ASSETS_DIR / "logo.png"), size="large")

BINARY_FEATURES = {"uses_https", "has_ip_address", "has_at_symbol"}


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner=False)
def _load_bundle(modified_time: float) -> dict:
    """Cached model load; the file's modified time busts the cache after retraining."""
    return load_model()


def get_bundle() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    try:
        return _load_bundle(MODEL_PATH.stat().st_mtime)
    except Exception as err:  # corrupt file, version mismatch...
        st.error(f"The saved model could not be loaded ({err}). Retrain it with `python -m src.train_model`.")
        return None


def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_training() -> None:
    with st.spinner("Training models - this can take a few minutes on the full dataset..."):
        try:
            train(pick_dataset(), verbose=False)
        except (FileNotFoundError, ValueError) as err:
            st.error(f"Training failed: {err}")
            return
    st.cache_resource.clear()
    st.rerun()


def require_model() -> dict:
    """Return the model bundle or show a first-run card and stop the page."""
    bundle = get_bundle()
    if bundle is not None:
        return bundle
    page_header("Set up the model", "ThreatLens needs a trained model before it can scan URLs.")
    html("""<div class="tl-card"><h4>No trained model found</h4>
    <p>Train from the command line: <code>python -m src.train_model</code></p>
    <p>Or train right here. Without the real dataset this uses the small synthetic sample, which is fine for a first look.</p></div>""")
    st.write("")
    if st.button("Train model now", type="primary"):
        run_training()
    st.stop()


def scan_and_store(url: str, bundle: dict) -> dict:
    result = predict_url(url, bundle)
    if result["ok"]:
        db.save_scan(result)
    return result


# ------------------------------------------------------------- Overview
def page_overview() -> None:
    page_header("Overview", "What you have scanned so far, and how the detection model is performing.")
    history = db.get_history()
    metrics = load_metrics()

    total = len(history)
    phishing = int((history["prediction"] == "Phishing").sum()) if total else 0
    high = int((history["risk_level"] == "High").sum()) if total else 0

    cols = st.columns(4)
    cols[0].markdown(kpi("Total scans", total, "stored locally"), unsafe_allow_html=True)
    cols[1].markdown(kpi("Flagged as phishing", phishing, f"{phishing / total * 100:.0f}% of scans" if total else "no scans yet", "bad"), unsafe_allow_html=True)
    cols[2].markdown(kpi("Judged legitimate", total - phishing, "model estimate, not a guarantee", "ok"), unsafe_allow_html=True)
    cols[3].markdown(kpi("High-risk URLs", high, "phishing probability 65% or more", "warn"), unsafe_allow_html=True)

    if total == 0:
        st.write("")
        html("""<div class="tl-card"><h4>No scans yet</h4>
        <p>Scan a URL to start building your dashboard, or run a handful of example URLs through the real model.</p></div>""")
        st.write("")
        left, right, _ = st.columns([1, 1, 3])
        with left:
            st.page_link(PAGES["scanner"], label="Open URL Scanner", icon=":material/travel_explore:")
        with right:
            if st.button("Run demo scans", help="Scores 10 example URLs with the trained model"):
                bundle = require_model()
                for url in DEMO_URLS:
                    scan_and_store(url, bundle)
                st.rerun()
    else:
        section_title("Scan activity")
        c1, c2, c3 = st.columns([1, 1, 1.3])
        with c1:
            st.caption("Verdicts")
            st.plotly_chart(charts.verdict_donut(phishing, total - phishing), width="stretch", config={"displayModeBar": False})
        with c2:
            st.caption("Risk levels")
            st.plotly_chart(charts.risk_bar(history["risk_level"].value_counts().to_dict()), width="stretch", config={"displayModeBar": False})
        with c3:
            st.caption("Scans per day")
            st.plotly_chart(charts.timeline_bar(history), width="stretch", config={"displayModeBar": False})

        section_title("Latest scans")
        recent = history.head(6)[["scanned_at", "url", "prediction", "risk_level", "confidence"]].copy()
        recent["confidence"] = (recent["confidence"] * 100).round(1)
        recent.columns = ["Time", "URL", "Verdict", "Risk", "Confidence (%)"]
        st.dataframe(recent, hide_index=True, width="stretch")

    section_title("Detection model")
    if metrics:
        best = metrics["best_model"]
        m = metrics["models"][best]
        left, right = st.columns([1, 1.2])
        with left:
            tag = chip("Sample data", "warn") if metrics["dataset"]["dataset_kind"] == "sample" else chip("Real dataset", "ok")
            html(f"""<div class="tl-card">{tag}{chip(best, "info")}
            <p style="margin-top:.8rem">F1-score <b style="color:#E7ECF6">{m['f1']:.3f}</b>, recall <b style="color:#E7ECF6">{m['recall']:.3f}</b>,
            precision <b style="color:#E7ECF6">{m['precision']:.3f}</b> on {metrics['dataset']['test_rows']:,} held-out URLs.</p>
            <p>The model reads the domain name as text plus 11 URL features (the http/https scheme is ignored). It never visits the site. Open <b>Model Performance</b> for the full comparison.</p></div>""")
            if metrics["dataset"]["dataset_kind"] == "sample":
                st.write("")
                sample_data_warning()
        with right:
            st.caption(f"What the {best} pays attention to")
            st.plotly_chart(charts.importance_bar(metrics["feature_importance"][best], 300), width="stretch", config={"displayModeBar": False})
    else:
        callout("No metrics file found. Train a model to see performance here.", "info")


# ---------------------------------------------------------- URL Scanner
def _use_example(url: str) -> None:
    st.session_state["scan_url"] = url
    st.session_state["auto_scan"] = True


def page_scanner() -> None:
    page_header("URL Scanner", "Paste a link to check it. ThreatLens analyzes the text of the URL only and never opens the page.")
    bundle = require_model()

    with st.form("scan_form"):
        st.text_input("URL", key="scan_url", placeholder="https://example.com/login", label_visibility="collapsed")
        submitted = st.form_submit_button("Scan URL", type="primary")

    st.caption("Try an example (the suspicious ones are invented and not real threats):")
    for col, (label, url) in zip(st.columns(len(EXAMPLE_URLS)), EXAMPLE_URLS.items()):
        col.button(label, key=f"ex_{label}", on_click=_use_example, args=(url,), width="stretch")

    if submitted or st.session_state.pop("auto_scan", False):
        result = scan_and_store(st.session_state.get("scan_url", ""), bundle)
        if result["ok"]:
            st.session_state["last_result"] = result
        else:
            st.session_state["last_result"] = None
            st.error(result["error"])

    result = st.session_state.get("last_result")
    if not result:
        return

    st.write("")
    left, right = st.columns([1.7, 1])
    with left:
        verdict_card(result)
    with right:
        st.plotly_chart(charts.gauge(result["phishing_probability"]), width="stretch", config={"displayModeBar": False})
        st.caption("Phishing probability (low below 35%, high from 65%)")

    domain_mode_note(result)
    section_title("URL anatomy")
    url_anatomy(result["url"])

    section_title("Why it scored this way")
    if result["indicators"]:
        indicator_rows(result["indicators"][:3])
        if len(result["indicators"]) > 3:
            st.caption(f"{len(result['indicators']) - 3} more indicator(s) in the full analysis.")
    else:
        indicator_rows([], result["positive_signals"][:3])
    st.page_link(PAGES["analysis"], label="Open full threat analysis", icon=":material/troubleshoot:")
    callout(result["disclaimer"], "info")
    if result["dataset_kind"] == "sample":
        callout("The active model was trained on the synthetic sample dataset, so this verdict is for demonstration only.", "warn")


# ------------------------------------------------------ Threat Analysis
def _format_feature(name: str, value: float) -> str:
    if name in BINARY_FEATURES:
        return "Yes" if value else "No"
    return f"{value:,.0f}"


def _compare_table(features: dict, profiles: dict) -> str:
    rows = []
    for name in FEATURE_NAMES:
        value, legit, phish = features[name], profiles["legitimate"][name], profiles["phishing"][name]
        if name in BINARY_FEATURES:
            legit_txt, phish_txt = f"{legit * 100:.0f}% yes", f"{phish * 100:.0f}% yes"
        else:
            legit_txt, phish_txt = f"{legit:,.1f}", f"{phish:,.1f}"
        d_legit, d_phish = abs(value - legit), abs(value - phish)
        if abs(d_legit - d_phish) < 1e-9:
            closer = chip("Equal", "info")
        elif d_phish < d_legit:
            closer = chip("Phishing", "bad")
        else:
            closer = chip("Legitimate", "ok")
        rows.append(f"<tr><td>{esc(FEATURE_LABELS[name])}</td><td class='num'>{esc(_format_feature(name, value))}</td>"
                    f"<td class='num'>{legit_txt}</td><td class='num'>{phish_txt}</td><td>{closer}</td></tr>")
    return ("<table class='tl-table'><tr><th>Feature</th><th>This URL</th><th>Typical legitimate</th>"
            "<th>Typical phishing</th><th>Closer to</th></tr>" + "".join(rows) + "</table>")


def page_analysis() -> None:
    page_header("Threat Analysis", "A closer look at why a URL was flagged or passed, and what you should check next.")
    bundle = require_model()

    urls = []
    last = st.session_state.get("last_result")
    if last:
        urls.append(last["url"])
    urls += [u for u in db.get_history(limit=50)["url"].tolist() if u not in urls]
    if not urls:
        html('<div class="tl-card"><h4>Nothing to analyze yet</h4><p>Scan a URL first, then come back for the detailed breakdown.</p></div>')
        st.write("")
        st.page_link(PAGES["scanner"], label="Open URL Scanner", icon=":material/travel_explore:")
        return

    chosen = st.selectbox("Choose a scan to analyze", urls, format_func=lambda u: (u[:95] + "...") if len(u) > 95 else u)
    result = predict_url(chosen, bundle)
    if not result["ok"]:
        st.error(result["error"])
        return

    left, right = st.columns([1.7, 1])
    with left:
        verdict_card(result)
    with right:
        st.plotly_chart(charts.gauge(result["phishing_probability"]), width="stretch", config={"displayModeBar": False})

    domain_mode_note(result)
    section_title("URL anatomy")
    url_anatomy(result["url"])

    left, right = st.columns([1, 1.1], gap="large")
    with left:
        section_title("Suspicious indicators")
        if result["indicators"]:
            indicator_rows(result["indicators"])
        else:
            callout("No rule-of-thumb red flags found in the URL text.", "info")
        section_title("Neutral or reassuring traits")
        indicator_rows([], result["positive_signals"] or ["None found"])
    with right:
        section_title("How this URL compares with training data")
        metrics = load_metrics()
        if metrics:
            html(_compare_table(result["features"], metrics["class_profiles"]))
            st.caption("Averages come from the training dataset. The model weighs all 12 features together, so a single row is not the verdict.")
        else:
            callout("Train a model to see how this URL compares with the training data.", "info")

    section_title("What to do next")
    html("""<div class="tl-card"><h4>Model prediction is not verified threat intelligence</h4>
    <p>ThreatLens only reads the text of the URL. It does not know whether the site is on a blocklist, how old the domain is, or what the page contains.</p>
    <p>To verify a link, check it with a reputation service such as
    <a href="https://transparencyreport.google.com/safe-browsing/search" target="_blank" style="color:#9CB4FF">Google Safe Browsing</a> or
    <a href="https://www.virustotal.com" target="_blank" style="color:#9CB4FF">VirusTotal</a>.
    If a message asks you to log in or pay, open the organization's website by typing its address yourself.</p></div>""")


# --------------------------------------------------------- Scan History
def page_history() -> None:
    page_header("Scan History", "Every URL you have scanned, stored in a local SQLite file on this computer.")
    history = db.get_history()
    if history.empty:
        html('<div class="tl-card"><h4>No scans yet</h4><p>Scanned URLs will appear here.</p></div>')
        st.write("")
        st.page_link(PAGES["scanner"], label="Open URL Scanner", icon=":material/travel_explore:")
        return

    c1, c2 = st.columns([1, 2])
    verdict = c1.radio("Verdict", ["All", "Phishing", "Legitimate"], horizontal=True)
    search = c2.text_input("Search URLs", placeholder="Filter by text in the URL")

    view = history
    if verdict != "All":
        view = view[view["prediction"] == verdict]
    if search:
        view = view[view["url"].str.contains(search, case=False, regex=False)]

    display = view[["scanned_at", "url", "prediction", "risk_level", "confidence", "phishing_probability"]].copy()
    display["confidence"] = (display["confidence"] * 100).round(1)
    display["phishing_probability"] = (display["phishing_probability"] * 100).round(1)
    display.columns = ["Time", "URL", "Verdict", "Risk", "Confidence (%)", "Phishing probability (%)"]
    st.caption(f"Showing {len(display)} of {len(history)} scans")
    st.dataframe(
        display, hide_index=True, width="stretch",
        column_config={
            "Confidence (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            "Phishing probability (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        },
    )
    st.download_button("Download as CSV", display.to_csv(index=False).encode("utf-8"), "threatlens_history.csv", "text/csv")

    with st.expander("Clear history"):
        st.write("This permanently deletes every stored scan.")
        confirmed = st.checkbox("I understand this cannot be undone")
        if st.button("Clear all history", disabled=not confirmed):
            removed = db.clear_history()
            st.session_state["last_result"] = None
            st.success(f"Deleted {removed} scans.")
            st.rerun()


# ---------------------------------------------------- Model Performance
def page_performance() -> None:
    page_header("Model Performance", "Metrics measured on held-out test URLs from the last training run. Nothing here is hardcoded.")
    require_model()
    metrics = load_metrics()
    if not metrics:
        callout("The model exists but metrics.json is missing. Retrain with `python -m src.train_model`.", "warn")
        return

    ds, models, best = metrics["dataset"], metrics["models"], metrics["best_model"]
    if ds["dataset_kind"] == "sample":
        sample_data_warning()
    if ds.get("training_mode") == "domain":
        callout(f"Domain mode: in this dataset {ds['homepage_share_legitimate'] * 100:.0f}% of legitimate URLs are bare homepages but only "
                f"{ds['homepage_share_phishing'] * 100:.0f}% of phishing URLs are. A model could score highly just by learning 'has a path = phishing'. "
                f"To avoid that, ThreatLens trained on homepage-style URLs only ({ds['rows_removed_with_paths']:,} URLs with paths were set aside) "
                "and scores the domain of any URL you scan. The metrics below measure domain-level detection on that subset.", "info")

    cols = st.columns(4)
    cols[0].markdown(kpi("Dataset", f"{ds['final_rows']:,}", f"{ds['source_file']}, after cleaning"), unsafe_allow_html=True)
    cols[1].markdown(kpi("Train / test split", f"{ds['train_rows']:,} / {ds['test_rows']:,}", "stratified, random_state 42"), unsafe_allow_html=True)
    cols[2].markdown(kpi("Best model", best, "highest F1-score", "ok"), unsafe_allow_html=True)
    cols[3].markdown(kpi("Trained", metrics["trained_at"].replace("T", " "), f"{ds['phishing_count']:,} phishing, {ds['legitimate_count']:,} legit"), unsafe_allow_html=True)

    section_title("Model comparison")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        rows = ""
        for name, m in models.items():
            badge = chip("Best", "ok") if name == best else ""
            auc = f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "n/a"
            rows += (f"<tr><td>{esc(name)} {badge}</td><td class='num'>{m['accuracy']:.3f}</td><td class='num'>{m['precision']:.3f}</td>"
                     f"<td class='num'>{m['recall']:.3f}</td><td class='num'>{m['f1']:.3f}</td><td class='num'>{auc}</td></tr>")
        html("<table class='tl-table'><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr>" + rows + "</table>")
        st.caption("Phishing is the positive class. Precision: of URLs flagged, how many were truly phishing. Recall: of all phishing URLs, how many were caught.")
    with right:
        st.plotly_chart(charts.model_comparison_bar(models), width="stretch", config={"displayModeBar": False})

    section_title("Confusion matrices")
    cm_cols = st.columns(len(models))
    for col, (name, m) in zip(cm_cols, models.items()):
        col.plotly_chart(charts.confusion_heatmap(m["confusion_matrix"], name), width="stretch", config={"displayModeBar": False})
    cm = models[best]["confusion_matrix"]
    callout(f"{best} on the test set: {cm['fp']} legitimate URL(s) wrongly flagged as phishing (false positives) and "
            f"{cm['fn']} phishing URL(s) missed (false negatives).", "info")

    left, right = st.columns(2, gap="large")
    with left:
        section_title("ROC curves")
        st.plotly_chart(charts.roc_chart(models), width="stretch", config={"displayModeBar": False})
    with right:
        section_title("Feature importance")
        choice = st.radio("Model", list(metrics["feature_importance"]), horizontal=True, index=list(metrics["feature_importance"]).index(best), label_visibility="collapsed")
        st.plotly_chart(charts.importance_bar(metrics["feature_importance"][choice]), width="stretch", config={"displayModeBar": False})

    with st.expander("Data cleaning report"):
        st.json({k: ds[k] for k in ("raw_rows", "missing_removed", "scheme_added", "duplicates_removed", "conflicting_removed", "final_rows")})
    with st.expander("Retrain the models"):
        st.write(f"Uses {pick_dataset().name}. Download the real dataset first (see README) to replace the sample.")
        if st.button("Retrain now"):
            run_training()


# ---------------------------------------------------- Security Awareness
def page_awareness() -> None:
    page_header("Security Awareness", "How phishing links work, and habits that protect you even when a detector gets it wrong.")

    section_title("Read a link before you click it")
    html("""<div class="tl-card"><p>The part that matters is the <b style="color:#E7ECF6">registered domain</b>, just before the first single slash.
    Everything to its left is controlled by whoever owns that domain.</p></div>""")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Trustworthy structure: the real domain is paypal.com")
        url_anatomy("https://www.paypal.com/signin", mark_keywords=False, show_legend=False)
    with c2:
        st.caption("Deceptive structure: the real domain is attacker-site.xyz")
        url_anatomy("http://paypal.com.secure-login.attacker-site.xyz/signin", mark_keywords=False)

    section_title("Common red flags")
    flags = [
        ("Urgency or threats", "Messages that say your account will be closed today are designed to stop you from thinking."),
        ("Look-alike domains", "Hyphens, extra words, or swapped letters, such as bank-secure-login.com or paypa1.com."),
        ("IP address links", "A link such as http://203.0.113.5/login is rarely a genuine service."),
        ("The @ trick", "In http://bank.com@evil.xyz the real destination is evil.xyz."),
        ("Unexpected attachments or QR codes", "A QR code can hide the full link. Preview it before opening."),
        ("Requests for passwords, OTPs or card details", "Legitimate organizations do not ask for these over email, SMS or chat."),
    ]
    html('<div class="tl-grid">' + "".join(f'<div class="tl-card"><h4>{esc(a)}</h4><p>{esc(b)}</p></div>' for a, b in flags) + "</div>")

    left, right = st.columns(2, gap="large")
    with left:
        section_title("If you already clicked")
        for text in ["Close the page. Do not enter or submit anything.",
                     "If you typed a password, change it right away and anywhere you reused it.",
                     "Turn on multi-factor authentication for important accounts.",
                     "Tell your bank immediately if you shared card or banking details.",
                     "Run a security scan on the device if you downloaded a file."]:
            indicator_rows([], [text])
    with right:
        section_title("Report phishing (India)")
        html("""<div class="tl-card">
        <p><b style="color:#E7ECF6">National Cyber Crime Reporting Portal:</b> cybercrime.gov.in</p>
        <p><b style="color:#E7ECF6">Cyber crime helpline:</b> 1930 (financial fraud)</p>
        <p><b style="color:#E7ECF6">CERT-In:</b> cert-in.org.in for incident reporting</p>
        <p>Also report the message to the impersonated organization and your email or messaging provider.</p></div>""")
        section_title("What ThreatLens cannot see")
        html("""<div class="tl-card"><p>Page content, domain age, SSL certificate details, blocklists, redirects, and links shortened by services such as bit.ly.
        Treat every result as one signal among several.</p></div>""")


# -------------------------------------------------------------- routing
PAGES = {
    "overview": st.Page(page_overview, title="Overview", icon=":material/space_dashboard:", url_path="overview", default=True),
    "scanner": st.Page(page_scanner, title="URL Scanner", icon=":material/travel_explore:", url_path="scanner"),
    "analysis": st.Page(page_analysis, title="Threat Analysis", icon=":material/troubleshoot:", url_path="analysis"),
    "history": st.Page(page_history, title="Scan History", icon=":material/history:", url_path="history"),
    "performance": st.Page(page_performance, title="Model Performance", icon=":material/monitoring:", url_path="performance"),
    "awareness": st.Page(page_awareness, title="Security Awareness", icon=":material/school:", url_path="awareness"),
}
navigation = st.navigation(list(PAGES.values()), position="sidebar")

with st.sidebar:
    _bundle = get_bundle()
    if _bundle:
        _kind = "synthetic sample data" if _bundle.get("dataset_kind") == "sample" else "real dataset"
        html(f'<div class="side-status"><span class="dot {"warn" if _bundle.get("dataset_kind") == "sample" else "ok"}"></span><b>{esc(_bundle["model_name"])}</b> active<br>Trained on {_kind}<br>URLs are analyzed as text and never visited.</div>')
    else:
        html('<div class="side-status"><span class="dot warn"></span><b>No model loaded</b><br>Train one to enable scanning.</div>')

navigation.run()
