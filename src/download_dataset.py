"""Download the real PhiUSIIL Phishing URL dataset from the UCI repository.

Dataset : PhiUSIIL Phishing URL (Website)  - 134,850 legitimate + 100,945 phishing URLs
Source  : https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
Paper   : Prasad & Chandra (2024), Computers & Security, doi:10.1016/j.cose.2023.103545
License : CC BY 4.0 (attribution required)

Usage:
    python -m src.download_dataset

If this fails (no internet, firewall), download the CSV manually - see README.md.
"""

from .config import DATA_DIR

OUTPUT_PATH = DATA_DIR / "phiusiil_urls.csv"
UCI_DATASET_ID = 967


def main() -> None:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise SystemExit("Missing package. Run: pip install ucimlrepo")

    print("Downloading PhiUSIIL dataset from UCI (about 55 MB, please wait)...")
    try:
        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    except Exception as err:  # network errors, UCI downtime, etc.
        raise SystemExit(
            f"Download failed: {err}\n"
            "Download the file manually instead (see 'Dataset' in README.md)."
        )

    features = dataset.data.features
    targets = dataset.data.targets
    if "URL" not in features.columns:
        raise SystemExit("The downloaded data has no 'URL' column; UCI may have changed the format.")

    # Keep only what we need. UCI label convention: 1 = legitimate, 0 = phishing.
    # (preprocessing.py detects this layout via the 'URLLength' column, so we keep it.)
    out = features[["URL", "URLLength"]].copy()
    out["label"] = targets.iloc[:, 0].values

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(out):,} rows to {OUTPUT_PATH}")
    print("Next step: python -m src.train_model")


if __name__ == "__main__":
    main()
