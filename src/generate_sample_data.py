"""Generate data/sample_urls.csv - a small SYNTHETIC dataset for smoke-testing.

These URLs are invented from templates (fixed random seed, so the output is
reproducible). They are NOT real threat data and model scores obtained from
them say nothing about real-world performance. Train on the real PhiUSIIL
dataset for meaningful results (see README.md).

Usage:  python -m src.generate_sample_data
"""

import random

import pandas as pd

from .config import SAMPLE_DATA_PATH

rng = random.Random(42)

BRANDS = ["paypal", "amazon", "microsoft", "netflix", "apple", "google", "hdfc", "sbi", "icici", "dropbox", "facebook", "instagram"]
LURE_WORDS = ["login", "secure", "verify", "update", "account", "signin", "confirm", "billing", "support", "unlock", "password", "wallet"]
SYLLABLES = ["ka", "lo", "mi", "ra", "ten", "vo", "zu", "bel", "nor", "sun", "dax", "pri", "qu", "fen", "gor", "lum"]
BAD_TLDS = [".xyz", ".top", ".info", ".club", ".online", ".site", ".click", ".work", ".link", ".buzz"]
GOOD_TLDS = [".com", ".org", ".net", ".edu", ".io", ".co.uk", ".in", ".gov"]
PLAIN_WORDS = ["garden", "travel", "recipes", "photo", "science", "books", "music", "health", "sports", "news", "learn", "market", "design", "cloud", "energy"]
REAL_LEGIT = [
    "https://www.wikipedia.org/", "https://github.com/", "https://www.python.org/", "https://www.mozilla.org/en-US/",
    "https://www.nasa.gov/", "https://www.bbc.com/news", "https://stackoverflow.com/questions", "https://www.khanacademy.org/",
    "https://docs.python.org/3/library/os.html", "https://www.reuters.com/world/", "https://www.who.int/", "https://www.iitb.ac.in/",
    "https://www.nytimes.com/section/technology", "https://developer.mozilla.org/en-US/docs/Web", "https://www.apple.com/in/",
    "https://www.microsoft.com/en-in/", "https://aws.amazon.com/getting-started/", "https://www.linkedin.com/jobs/",
    "http://www.example.org/", "http://neverssl.com/",
]


def fake_domain() -> str:
    return "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 4)))


def rand_id(n: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def phishing_url() -> str:
    scheme = rng.choice(["http", "http", "https"])  # many real phishing sites use HTTPS too
    kind = rng.randint(0, 5)
    brand, lure, tld = rng.choice(BRANDS), rng.choice(LURE_WORDS), rng.choice(BAD_TLDS)
    if kind == 0:   # brand + lure words glued together with hyphens
        return f"{scheme}://{brand}-{lure}-{rng.choice(LURE_WORDS)}.{fake_domain()}{tld}/{lure}/{rand_id(8)}.php"
    if kind == 1:   # raw IP address (documentation ranges, not real hosts)
        ip = rng.choice(["192.0.2", "198.51.100", "203.0.113"]) + f".{rng.randint(1, 254)}"
        return f"http://{ip}/{rng.choice(LURE_WORDS)}/{rng.choice(['index', 'update', 'login'])}.html?id={rng.randint(10000, 999999)}"
    if kind == 2:   # @ trick
        return f"{scheme}://www.{brand}.com@{fake_domain()}{tld}/{lure}?session={rand_id(12)}"
    if kind == 3:   # many subdomains
        return f"{scheme}://{lure}.{rng.choice(LURE_WORDS)}.{brand}.com.{fake_domain()}{tld}/{rand_id(6)}/{lure}"
    if kind == 4:   # long path with digits
        return f"{scheme}://{fake_domain()}-{brand}{tld}/wp-content/{rand_id(10)}/{lure}-{rng.randint(1000, 99999)}/{rand_id(16)}.html"
    # kind 5: quieter lookalike with fewer obvious signals
    return f"{scheme}://{brand}{rng.randint(1, 99)}.{fake_domain()}{tld}/{lure}"


def legit_url() -> str:
    kind = rng.randint(0, 5)
    word, tld = rng.choice(PLAIN_WORDS), rng.choice(GOOD_TLDS)
    scheme = rng.choice(["https", "https", "https", "http"])
    if kind == 0:
        return f"{scheme}://www.{word}{fake_domain()}{tld}/"
    if kind == 1:
        return f"{scheme}://{fake_domain()}{word}{tld}/{rng.choice(PLAIN_WORDS)}/{rng.choice(PLAIN_WORDS)}-{rng.choice(PLAIN_WORDS)}"
    if kind == 2:   # news-style article URL with a date and long slug
        return f"{scheme}://www.{fake_domain()}news{tld}/2024/{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.choice(PLAIN_WORDS)}-{rng.choice(PLAIN_WORDS)}-{rng.choice(PLAIN_WORDS)}-{rng.choice(PLAIN_WORDS)}.html"
    if kind == 3:   # legitimate sites also have login / account pages
        return f"{scheme}://accounts.{fake_domain()}{word}{tld}/login"
    if kind == 4:
        return f"{scheme}://docs.{fake_domain()}{tld}/{rng.choice(PLAIN_WORDS)}/guide?page={rng.randint(1, 40)}&lang=en"
    return f"{scheme}://support.{fake_domain()}{word}{tld}/help/{rng.choice(PLAIN_WORDS)}"


def main() -> None:
    rows = [(u, "legitimate") for u in REAL_LEGIT]
    rows += [(legit_url(), "legitimate") for _ in range(180)]
    rows += [(phishing_url(), "phishing") for _ in range(200)]
    df = pd.DataFrame(rows, columns=["url", "label"]).drop_duplicates("url")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(SAMPLE_DATA_PATH, index=False)
    print(f"Wrote {len(df)} synthetic URLs to {SAMPLE_DATA_PATH}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
