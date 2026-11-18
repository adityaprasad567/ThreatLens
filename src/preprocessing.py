"""Dataset loading, cleaning and feature-matrix building.

Internal label convention used everywhere in ThreatLens:
    is_phishing = 1  -> phishing
    is_phishing = 0  -> legitimate

Supported dataset layouts:
  * PhiUSIIL (UCI):  columns 'URL' and 'label' where label 1 = LEGITIMATE and 0 = PHISHING
                     (detected automatically and flipped to our convention).
  * Generic CSV:     a URL column plus a label column that is either text
                     (phishing / legitimate / malicious / benign / bad / good ...)
                     or numeric where 1 = phishing and 0 = legitimate.
"""

from pathlib import Path

import pandas as pd

from .config import SAMPLE_DATA_PATH
from urllib.parse import urlparse

from .feature_extraction import extract_feature_frame, normalize_scheme, to_domain_url

URL_COLUMNS = ("url", "URL", "Url", "link", "Link")
LABEL_COLUMNS = ("label", "label_legit_1_phish_0", "Label", "type", "Type", "status", "Status", "class", "Class", "result", "Result")

PHISHING_WORDS = {"phishing", "phish", "malicious", "malware", "bad"}
LEGIT_WORDS = {"legitimate", "legit", "benign", "good", "safe"}

MIN_ROWS_PER_CLASS = 20


def _find_column(columns, candidates) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(
        f"Could not find a column named one of {list(candidates)}. "
        f"The file has these columns: {list(columns)[:15]}"
    )


def _map_labels(series: pd.Series, phiusiil_layout: bool) -> pd.Series:
    """Convert any supported label format to is_phishing (1/0, NaN if unknown)."""
    if phiusiil_layout:
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.map({0: 1, 1: 0})  # PhiUSIIL: 1 = legitimate, 0 = phishing

    def convert(value):
        text = str(value).strip().lower()
        if text in PHISHING_WORDS or text in {"1", "1.0"}:
            return 1
        if text in LEGIT_WORDS or text in {"0", "0.0"}:
            return 0
        return float("nan")

    return series.map(convert)


def load_and_clean(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read a dataset CSV and return (clean_df, report).

    clean_df has two columns: 'url' and 'is_phishing'.
    The report records every cleaning step so nothing is hidden.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    url_col = _find_column(header, URL_COLUMNS)
    label_col = _find_column(header, LABEL_COLUMNS)
    phiusiil_layout = "URLLength" in header or "FILENAME" in header or label_col == "label_legit_1_phish_0"  # 1 = legitimate

    raw = pd.read_csv(
        path,
        usecols=[url_col, label_col],
        encoding="utf-8-sig",
        encoding_errors="replace",
        dtype={url_col: "string"},
    )
    df = pd.DataFrame(
        {"url": raw[url_col], "is_phishing": _map_labels(raw[label_col], phiusiil_layout)}
    )
    report = {"source_file": path.name, "raw_rows": int(len(df))}

    # 1. missing / empty values (includes labels we could not interpret)
    df["url"] = df["url"].astype("string").str.strip()
    before = len(df)
    df = df.dropna(subset=["url", "is_phishing"])
    df = df[df["url"] != ""]
    report["missing_removed"] = int(before - len(df))

    # URLs without a scheme (some datasets omit it) -> assume http://
    no_scheme = ~df["url"].str.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
    report["scheme_added"] = int(no_scheme.sum())
    df.loc[no_scheme, "url"] = "http://" + df.loc[no_scheme, "url"]

    # 2. exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["url", "is_phishing"])
    report["duplicates_removed"] = int(before - len(df))

    # 3. the same URL labelled both ways -> unreliable, drop every copy
    before = len(df)
    df = df[~df["url"].duplicated(keep=False)]
    report["conflicting_removed"] = int(before - len(df))

    df = df.reset_index(drop=True)
    df["is_phishing"] = df["is_phishing"].astype(int)
    df["url"] = df["url"].astype(str)

    counts = df["is_phishing"].value_counts()
    report["final_rows"] = int(len(df))
    report["phishing_count"] = int(counts.get(1, 0))
    report["legitimate_count"] = int(counts.get(0, 0))
    report["dataset_kind"] = "sample" if path.resolve() == SAMPLE_DATA_PATH.resolve() else "real"

    if min(report["phishing_count"], report["legitimate_count"]) < MIN_ROWS_PER_CLASS:
        raise ValueError(
            "After cleaning, each class needs at least "
            f"{MIN_ROWS_PER_CLASS} URLs. Found {report['phishing_count']} phishing and "
            f"{report['legitimate_count']} legitimate."
        )
    return df, report


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Turn the clean URL table into (X, y) using the shared feature extractor."""
    X = extract_feature_frame(df["url"].map(normalize_scheme))   # model never sees http vs https
    X["uses_https"] = df["url"].str.lower().str.startswith("https://").astype(int).to_numpy()  # for UI only
    y = df["is_phishing"].reset_index(drop=True)
    return X, y


# ------------------------------------------------- dataset shortcut check
HOMEPAGE_SHORTCUT_THRESHOLD = 0.95


def _is_homepage(url: str) -> bool:
    """True if the URL has no path, query or fragment (e.g. https://example.com or https://example.com/)."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.path in ("", "/") and not parsed.query and not parsed.fragment


def prepare_for_training(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, str]:
    """Detect the 'legitimate URLs are all bare homepages' shortcut and neutralise it.

    If (almost) every legitimate URL is a homepage, a model can score ~100% by simply
    learning "has a path = phishing". That says nothing about phishing behaviour, and
    it flags every real page with a path. In that case we train in DOMAIN mode:
    only homepage-style URLs are kept (both classes), so the model compares like with like
    and judges the domain only. Returns (df, mode) where mode is "domain" or "full".
    """
    homepage = df["url"].map(_is_homepage)
    legit_share = float(homepage[df["is_phishing"] == 0].mean())
    phish_share = float(homepage[df["is_phishing"] == 1].mean())
    report["homepage_share_legitimate"] = round(legit_share, 4)
    report["homepage_share_phishing"] = round(phish_share, 4)

    if legit_share < HOMEPAGE_SHORTCUT_THRESHOLD:
        report["training_mode"] = "full"
        report["rows_removed_with_paths"] = 0
        return df, "full"

    kept = df[homepage].copy()
    kept["url"] = kept["url"].map(to_domain_url)
    kept = kept.drop_duplicates(subset=["url", "is_phishing"])
    kept = kept[~kept["url"].duplicated(keep=False)].reset_index(drop=True)
    report["training_mode"] = "domain"
    report["rows_removed_with_paths"] = int(len(df) - len(kept))
    report["final_rows"] = int(len(kept))
    report["phishing_count"] = int((kept["is_phishing"] == 1).sum())
    report["legitimate_count"] = int((kept["is_phishing"] == 0).sum())
    if min(report["phishing_count"], report["legitimate_count"]) < MIN_ROWS_PER_CLASS:
        raise ValueError("Too few homepage-style URLs remain to train in domain mode.")
    return kept, "domain"


# ------------------------------------------------- extra legitimate examples
def augment_with_top_domains(df: pd.DataFrame, report: dict, n: int = 40000, seed: int = 42) -> pd.DataFrame:
    """Add legitimate homepage URLs from the popular-domain list.

    The PhiUSIIL legitimate URLs are 100% https and 100% 'www.', so a model trained on
    them alone calls everything else phishing. These rows add http, non-www and
    subdomain variants of well-known domains so the model cannot use those as shortcuts.
    """
    import random

    from .trusted import load_top_domains

    rng = random.Random(seed)
    domains = sorted(load_top_domains())
    if not domains:
        return df
    rng.shuffle(domains)
    subs = ["docs", "mail", "en", "blog", "support", "api", "data", "app", "news", "shop", "learn", "help"]
    rows = []
    for d in domains[:n]:
        r = rng.random()
        if r < 0.55:
            url = f"https://{d}"
        elif r < 0.80:
            url = f"http://{d}"
        else:
            url = f"{rng.choice(['https', 'https', 'http'])}://{rng.choice(subs)}.{d}"
        rows.append((url, 0))
    extra = pd.DataFrame(rows, columns=["url", "is_phishing"])
    out = pd.concat([df, extra], ignore_index=True).drop_duplicates(subset=["url"]).reset_index(drop=True)
    report["augmented_legit_rows"] = int(len(out) - len(df))
    report["final_rows"] = int(len(out))
    report["legitimate_count"] = int((out["is_phishing"] == 0).sum())
    return out
