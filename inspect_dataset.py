"""Inspect a phishing dataset CSV: columns, label values, and what each label looks like.

Usage (project root):  python inspect_dataset.py data/PhiLegit-100K.csv
Paste the output back; it never prints whole rows.
"""

import sys
from urllib.parse import urlparse

import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "data/PhiLegit-100K.csv"
header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
print("Columns:", header[:12], "..." if len(header) > 12 else "")

lower = {c.lower(): c for c in header}
url_col = next((lower[k] for k in ("url", "link") if k in lower), None)
label_col = next((lower[k] for k in ("label", "type", "status", "class", "result") if k in lower), None)
source_col = lower.get("source_dataset")
if not url_col or not label_col:
    sys.exit(f"Could not find URL/label columns (found url={url_col}, label={label_col}).")

use = [c for c in (url_col, label_col, source_col) if c]
df = pd.read_csv(path, usecols=use, encoding="utf-8-sig", encoding_errors="replace")
df = df.rename(columns={url_col: "url", label_col: "label"})
print(f"\nRows: {len(df):,}   URL column: {url_col}   Label column: {label_col}")
print("\nLabel values:\n" + df["label"].value_counts(dropna=False).to_string())
if source_col:
    print("\nLabel x source_dataset:\n" + pd.crosstab(df[source_col], df["label"]).to_string())


def parts(url):
    try:
        p = urlparse(str(url))
        return pd.Series({"https": p.scheme == "https", "homepage": p.path in ("", "/") and not p.query and not p.fragment})
    except ValueError:
        return pd.Series({"https": False, "homepage": False})


df = df.join(df["url"].apply(parts))
print("\nPer label (https share, homepage share, mean URL length):")
summary = df.groupby("label").agg(rows=("url", "size"), https=("https", "mean"), homepage=("homepage", "mean"), length=("url", lambda s: s.astype(str).str.len().mean()))
print(summary.round(3).to_string())

for value, group in df.groupby("label"):
    print(f"\n8 sample URLs with label = {value!r}:")
    print(group["url"].sample(min(8, len(group)), random_state=1).astype(str).str.slice(0, 90).to_string(index=False))
