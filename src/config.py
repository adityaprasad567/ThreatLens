"""Central configuration for ThreatLens: file paths and tunable constants."""

from pathlib import Path

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

SAMPLE_DATA_PATH = DATA_DIR / "sample_urls.csv"      # small synthetic file (demo only)
DB_PATH = DATA_DIR / "threatlens.db"                 # SQLite scan history
MODEL_PATH = MODELS_DIR / "best_model.joblib"        # saved best model
METRICS_PATH = MODELS_DIR / "metrics.json"           # real metrics from the last training run

# Real dataset files we look for automatically (see README for download steps)
REAL_DATASET_CANDIDATES = [
    DATA_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv",  # manual download from UCI
    DATA_DIR / "phiusiil_urls.csv",                  # created by src/download_dataset.py
]

# ---------------------------------------------------------------- training
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------- scanning
MAX_URL_LENGTH = 2048

# Risk level is derived from the model's phishing probability.
RISK_LOW_MAX = 0.35     # below this  -> Low
RISK_HIGH_MIN = 0.65    # at/above    -> High   (in between -> Medium)

# Example URLs for the scanner. The "suspicious" ones are made up and
# point to reserved/non-existent hosts - they are NOT real threats.
EXAMPLE_URLS = {
    "Legitimate-looking": "https://www.wikipedia.org/",
    "Suspicious login": "http://paypal-secure-login.account-verify.xyz/webscr/login.php?id=83921",
    "IP address host": "http://192.0.2.45/bank/update-password.html",
    "@ trick": "http://www.google.com@login-check.top/verify/account",
}

# URLs used by the "Run demo scans" button so the dashboard has content.
DEMO_URLS = [
    "https://www.wikipedia.org/",
    "https://github.com/",
    "https://docs.python.org/3/library/os.html",
    "https://www.bbc.com/news",
    "https://www.nasa.gov/",
    "http://paypal-secure-login.account-verify.xyz/webscr/login.php?id=83921",
    "http://192.0.2.45/bank/update-password.html",
    "http://www.google.com@login-check.top/verify/account",
    "http://secure.login.microsoft.com.auth-update.top/signin/confirm?user=1928374",
    "https://netflix-billing-support.info/account/suspended/unlock",
]
