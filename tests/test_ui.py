"""Headless UI tests using Streamlit's AppTest.

AppTest cannot click through st.navigation, so each test renders ONE page by
running app.py with `navigation.run()` swapped for a direct call to that page.
These tests use the real model/database files in the project, so train a model
first (python -m src.train_model) and expect the scan history to be modified.

Run from the project root:  python -m pytest tests/test_ui.py -v
"""

import sys
import warnings
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, database as db  # noqa: E402

warnings.filterwarnings("ignore")
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="Train a model first: python -m src.train_model")


def render(page: str) -> AppTest:
    """Run app.py but display only the named page function (e.g. 'scanner')."""
    source = f"import sys; sys.path.insert(0, {str(ROOT)!r})\n" + APP_SOURCE.replace(
        "navigation.run()", f"page_{page}()"
    )
    return AppTest.from_string(source, default_timeout=90).run()


@pytest.fixture(autouse=True)
def clean_history():
    db.clear_history()
    yield


@pytest.mark.parametrize("page", ["overview", "scanner", "analysis", "history", "performance", "awareness"])
def test_every_page_renders_without_errors(page):
    at = render(page)
    assert not at.exception, [e.value for e in at.exception]


def test_scanner_flags_suspicious_url_and_saves_it():
    at = render("scanner")
    at.text_input(key="scan_url").set_value("http://paypal-secure-login.account-verify.xyz/webscr/login.php?id=83921")
    [b for b in at.button if b.label == "Scan URL"][0].click().run()
    assert not at.exception
    assert db.count_scans() == 1
    row = db.get_history().iloc[0]
    assert row["prediction"] == "Phishing" and row["risk_level"] in ("Medium", "High")


def test_scanner_accepts_legitimate_url():
    at = render("scanner")
    at.text_input(key="scan_url").set_value("https://www.wikipedia.org/")
    [b for b in at.button if b.label == "Scan URL"][0].click().run()
    assert not at.exception
    assert db.get_history().iloc[0]["prediction"] == "Legitimate"


@pytest.mark.parametrize("bad", ["example.com", "not a url", "ftp://x.com", ""])
def test_scanner_rejects_invalid_urls_without_saving(bad):
    at = render("scanner")
    at.text_input(key="scan_url").set_value(bad)
    [b for b in at.button if b.label == "Scan URL"][0].click().run()
    assert not at.exception
    assert len(at.error) == 1
    assert db.count_scans() == 0


def test_history_page_clear_button():
    for url in ("https://www.wikipedia.org/", "http://192.0.2.45/login"):
        from src.prediction import predict_url
        db.save_scan(predict_url(url))
    at = render("history")
    assert not at.exception
    at.checkbox[0].check().run()
    [b for b in at.button if b.label == "Clear all history"][0].click().run()
    assert db.count_scans() == 0
