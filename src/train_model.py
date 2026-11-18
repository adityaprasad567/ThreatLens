
"""Reproducible domain-grouped ThreatLens training.

Usage:
    python -m src.train_model --data data/Combined_PhiUSIIL_PhiLegit.csv

Features:
- Complete URL character n-grams
- URL lexical features
- Domain features
- URL path and query features
- Domain-grouped train/test splitting
"""

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from .config import (
    MODEL_PATH,
    METRICS_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

from .feature_extraction import (
    MODEL_FEATURE_NAMES,
    normalize_scheme,
)

from .preprocessing import (
    load_and_clean,
    build_feature_matrix,
)


TEXT_COLUMN = "host_text"

TWO_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk",
    "co.in", "org.in", "net.in", "ac.in",
    "gov.in", "edu.in", "com.au", "net.au",
    "org.au", "edu.au", "co.jp", "co.nz",
    "co.za", "com.br", "com.cn", "com.mx",
    "com.sg", "com.tr", "com.pk",
}


# --------------------------------------------------
# FIND DATASET
# --------------------------------------------------

def pick_dataset():
    """Return the preferred available dataset."""

    project_root = Path(__file__).resolve().parent.parent

    candidates = [
        project_root / "data" / "Combined_PhiUSIIL_PhiLegit.csv",
        project_root / "data" / "PhiUSIIL_original.csv",
        project_root / "data" / "PhiLegit-100K.csv",
    ]

    for path in candidates:
        if path.is_file():
            return path

    return None


# --------------------------------------------------
# REGISTERED DOMAIN EXTRACTION
# --------------------------------------------------

def registrable_domain(url: str) -> str:
    """Extract an approximate registered domain."""

    try:
        normalized = normalize_scheme(str(url))
        host = (
            urlparse(normalized).hostname or ""
        ).lower().rstrip(".")

    except ValueError:
        return str(url)

    labels = host.split(".")

    if len(labels) < 2:
        return host

    suffix = ".".join(labels[-2:])

    n = 3 if suffix in TWO_PART_SUFFIXES else 2

    return (
        ".".join(labels[-n:])
        if len(labels) >= n
        else host
    )


# --------------------------------------------------
# DOMAIN-GROUPED DATA SPLITTING
# --------------------------------------------------

def grouped_split(
    X,
    y,
    groups,
    test_size=TEST_SIZE,
    seed=RANDOM_STATE,
):
    """Create a deterministic domain-grouped split."""

    best = None
    target = float(y.mean())

    for rs in range(seed, seed + 40):

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=rs,
        )

        tr, te = next(
            splitter.split(X, y, groups)
        )

        if (
            y.iloc[tr].nunique() < 2
            or y.iloc[te].nunique() < 2
        ):
            continue

        score = (
            abs(len(te) / len(y) - test_size)
            + abs(float(y.iloc[te].mean()) - target)
        )

        if best is None or score < best[0]:
            best = (score, tr, te, rs)

    if best is None:
        raise ValueError(
            "Could not create a valid domain-grouped split."
        )

    return best[1], best[2], best[3]


# --------------------------------------------------
# TRAIN MACHINE LEARNING MODEL
# --------------------------------------------------

def train(data_path=None):
    """Train and save the ThreatLens phishing model."""

    if data_path is None:
        data_path = pick_dataset()

    if data_path is None:
        raise FileNotFoundError(
            "Training dataset not found in the data folder."
        )

    data_path = Path(data_path)

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    print(f"Loading {data_path} ...")

    df, report = load_and_clean(data_path)

    X, y = build_feature_matrix(df)

    url_text = (
        df.url
        .map(normalize_scheme)
        .str.lower()
        .reset_index(drop=True)
    )

    groups = (
        df.url
        .map(registrable_domain)
        .reset_index(drop=True)
    )

    tr, te, split_seed = grouped_split(
        X, y, groups
    )

    Xtr = X.iloc[tr][MODEL_FEATURE_NAMES].copy()
    Xte = X.iloc[te][MODEL_FEATURE_NAMES].copy()

    Xtr[TEXT_COLUMN] = url_text.iloc[tr].to_numpy()
    Xte[TEXT_COLUMN] = url_text.iloc[te].to_numpy()

    ytr = y.iloc[tr]
    yte = y.iloc[te]

    model = Pipeline([
        (
            "features",
            ColumnTransformer([
                (
                    "url_char_ngrams",
                    HashingVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        n_features=2**18,
                        alternate_sign=False,
                        norm="l2",
                        dtype=np.float32,
                    ),
                    TEXT_COLUMN,
                ),
                (
                    "lexical_domain_numeric",
                    StandardScaler(),
                    MODEL_FEATURE_NAMES,
                ),
            ], remainder="drop"),
        ),
        (
            "clf",
            SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                max_iter=12,
                tol=1e-3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                early_stopping=True,
                n_iter_no_change=3,
                validation_fraction=0.1,
            ),
        ),
    ])

    print(
        f"Training on {len(tr):,} URLs; "
        f"testing on {len(te):,} URLs "
        f"({groups.iloc[te].nunique():,} unseen domains) ..."
    )

    model.fit(Xtr, ytr)

    pred = model.predict(Xte)

    phishing_class_index = list(
        model.classes_
    ).index(1)

    prob = model.predict_proba(Xte)[
        :, phishing_class_index
    ]

    tn, fp, fn, tp = confusion_matrix(
        yte,
        pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(
            precision_score(yte, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(yte, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(yte, pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(yte, prob)
        ),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "false_negative_rate": (
            float(fn / (fn + tp))
            if fn + tp
            else None
        ),
        "classification_report": classification_report(
            yte,
            pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trained_at = datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")

    bundle = {
        "model": model,
        "model_name": (
            "Hashed Character N-gram + "
            "URL lexical/domain features (SGD logistic)"
        ),
        "feature_names": MODEL_FEATURE_NAMES,
        "uses_host_text": True,
        "text_input": "complete_normalized_url",
        "dataset_kind": "real",
        "training_mode": "full_url_domain_grouped",
        "trained_at": trained_at,
        "label_convention": (
            "1=phishing, 0=legitimate"
        ),
        "split_strategy": (
            "GroupShuffleSplit by approximate "
            "registrable domain"
        ),
        "domain_extractor": (
            "built-in common two-part suffix list; "
            "not a complete Public Suffix List"
        ),
    }

    joblib.dump(
        bundle,
        MODEL_PATH,
        compress=3,
    )

    sha = hashlib.sha256(
        data_path.read_bytes()
    ).hexdigest()

    output = {
        "trained_at": trained_at,
        "model_name": bundle["model_name"],
        "best_model": bundle["model_name"],

        "dataset": {
            "source_file": data_path.name,
            "sha256": sha,
            "raw_rows": report["raw_rows"],
            "clean_rows": len(df),
            "duplicates_removed": report[
                "duplicates_removed"
            ],
            "conflicting_removed": report[
                "conflicting_removed"
            ],
            "missing_removed": report[
                "missing_removed"
            ],
            "phishing_count": int(y.sum()),
            "legitimate_count": int((y == 0).sum()),
            "train_rows": len(tr),
            "test_rows": len(te),
            "test_size": len(te) / len(df),
            "unique_domains": int(groups.nunique()),
            "train_unique_domains": int(
                groups.iloc[tr].nunique()
            ),
            "test_unique_domains": int(
                groups.iloc[te].nunique()
            ),
            "domain_overlap": int(
                len(
                    set(groups.iloc[tr])
                    & set(groups.iloc[te])
                )
            ),
            "split_seed": split_seed,
            "train_phishing_rate": float(ytr.mean()),
            "test_phishing_rate": float(yte.mean()),
        },

        "features": {
            "text": (
                "complete normalized URL hashed "
                "character n-grams (3-5)"
            ),
            "numeric": MODEL_FEATURE_NAMES,
            "includes_hostname": True,
            "includes_path": True,
            "includes_query": True,
        },

        "metrics": metrics,

        "reproducibility": {
            "random_state": RANDOM_STATE,
            "test_size_target": TEST_SIZE,
            "python_command": (
                "python -m src.train_model "
                f"--data {data_path.name}"
            ),
            "model_file": "models/best_model.joblib",
        },
    }

    METRICS_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))

    print(
        "Domain overlap:",
        output["dataset"]["domain_overlap"],
    )

    print(
        f"Saved {MODEL_PATH} and {METRICS_PATH}"
    )

    return output


# --------------------------------------------------
# COMMAND LINE ENTRY POINT
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=None,
        help="Path to training CSV",
    )

    args = parser.parse_args()

    train(args.data)


if __name__ == "__main__":
    main()
