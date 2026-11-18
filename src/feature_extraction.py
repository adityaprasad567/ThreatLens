"""URL validation and lexical feature extraction.

IMPORTANT: this single module is used for BOTH training and prediction, so the
features are always computed in exactly the same way. ThreatLens only looks at
the URL text - it never connects to the website.
"""

import ipaddress
import re
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd

from .config import MAX_URL_LENGTH

# Order matters: the model is trained on columns in exactly this order.
FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_hyphens",
    "num_subdomains",
    "uses_https",
    "has_ip_address",
    "has_at_symbol",
    "num_special_chars",
    "suspicious_keyword_count",
    "path_length",
    "num_digits",
]

# Features the MODEL is trained on. "uses_https" is deliberately left out: in the public datasets
# almost every legitimate URL is https while many phishing URLs are http, so the model turned the
# scheme into a shortcut and flagged any http:// version of a normal site. HTTPS is still computed
# and shown as a rule-based indicator in the UI, it just no longer drives the score.
MODEL_FEATURE_NAMES = [f for f in FEATURE_NAMES if f != "uses_https"]

# Human-friendly names for the UI.
FEATURE_LABELS = {
    "url_length": "URL length",
    "domain_length": "Domain length",
    "num_dots": "Number of dots",
    "num_hyphens": "Number of hyphens",
    "num_subdomains": "Number of subdomains",
    "uses_https": "Uses HTTPS",
    "has_ip_address": "IP address as host",
    "has_at_symbol": "Contains @ symbol",
    "num_special_chars": "Special characters",
    "suspicious_keyword_count": "Suspicious keywords",
    "path_length": "Path length",
    "num_digits": "Number of digits",
    "domain_text_patterns": "Domain name patterns (n-grams)",
}

# Words that often appear in credential-phishing URLs. Legitimate sites use
# some of them too, so this is a signal for the model - not proof.
SUSPICIOUS_KEYWORDS = (
    "login", "signin", "verify", "update", "secure", "account", "banking",
    "confirm", "password", "paypal", "webscr", "wallet", "suspend", "unlock",
    "invoice", "billing", "support", "alert", "credential", "authenticate",
    "recover", "validate", "prize", "claim", "winner", "reward", "gift", "bonus",
)

# Registered domains that sit under a two-part public suffix (e.g. example.co.uk).
# A tiny hand-made list keeps us dependency-free; it is not exhaustive.
TWO_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "co.in", "org.in", "net.in", "ac.in",
    "gov.in", "edu.in", "com.au", "net.au", "org.au", "edu.au", "co.jp",
    "co.nz", "co.za", "com.br", "com.cn", "com.mx", "com.sg", "com.tr", "com.pk",
}

_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{1,62}$"
)


# --------------------------------------------------------------- helpers
def is_ip_host(host: str) -> bool:
    """True for IPv4/IPv6 literals and for obfuscated forms such as 3232235777 or 0xC0A80001."""
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if host.isdigit() and int(host) < 2**32:
        return True
    return bool(re.fullmatch(r"0x[0-9a-f]{1,8}", host, re.IGNORECASE))


def split_host(host: str) -> tuple[str, str]:
    """Split a hostname into (subdomain_part, registered_domain).

    'login.secure.example.co.uk' -> ('login.secure', 'example.co.uk')
    """
    host = host.lower()
    labels = host.split(".")
    if is_ip_host(host) or len(labels) <= 2:
        return "", host
    suffix_len = 2 if ".".join(labels[-2:]) in TWO_PART_SUFFIXES else 1
    registered_len = suffix_len + 1
    if len(labels) <= registered_len:
        return "", host
    return ".".join(labels[:-registered_len]), ".".join(labels[-registered_len:])


def count_subdomains(host: str) -> int:
    """Number of subdomain labels, ignoring a leading 'www'."""
    sub, _ = split_host(host)
    if not sub:
        return 0
    labels = sub.split(".")
    return len(labels) - (1 if labels[0] == "www" else 0)


def find_suspicious_keywords(url: str) -> list[str]:
    """Return the distinct suspicious keywords found in the URL (lower-case)."""
    lowered = url.lower()
    return [kw for kw in SUSPICIOUS_KEYWORDS if kw in lowered]


def to_domain_url(url: str) -> str:
    """'https://a.com/x?y=1' -> 'https://a.com' (keeps any user@ part so the @ trick stays visible)."""
    url = url.strip()
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    if not parsed.scheme or not parsed.netloc:
        return url
    return f"{parsed.scheme}://{parsed.netloc}"


# ------------------------------------------------------------ validation
def validate_url(raw_url: str | None) -> tuple[bool, str, str]:
    """Check that a string looks like a scannable http(s) URL.

    Returns (is_valid, cleaned_url, message). The URL is never opened.
    """
    if raw_url is None or not raw_url.strip():
        return False, "", "Enter a URL to scan."
    url = raw_url.strip()
    if len(url) > MAX_URL_LENGTH:
        return False, url, f"This URL is longer than {MAX_URL_LENGTH} characters."
    if re.search(r"\s", url):
        return False, url, "A URL cannot contain spaces. Remove them and try again."
    if not re.match(r"^https?://", url, re.IGNORECASE):
        return False, url, (
            "Start the URL with http:// or https:// "
            "(for example https://example.com/login)."
        )
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        _ = parsed.port  # raises ValueError for an invalid port
    except ValueError:
        return False, url, "This URL is malformed (check the brackets and port number)."
    if not host:
        return False, url, "The URL has no domain name after http:// or https://."
    if not is_ip_host(host):
        try:
            ascii_host = host.encode("idna").decode("ascii")
        except UnicodeError:
            return False, url, "The domain name contains characters that are not valid."
        if not _DOMAIN_RE.match(ascii_host.lower()):
            return False, url, (
                "The domain name looks invalid. It should look like example.com."
            )
    return True, url, "OK"


def strip_www(url: str) -> str:
    """'https://www.a.com/x' -> 'https://a.com/x'.

    In the PhiUSIIL data every legitimate URL starts with 'www.', so without this
    the model learns 'no www = phishing' and flags github.com, docs.python.org, etc.
    """
    return re.sub(r"^([a-zA-Z][a-zA-Z0-9+.-]*://)www\.", r"\1", url.strip(), count=1)


# ------------------------------------------------------ feature extraction
def extract_features(url: str) -> dict[str, float]:
    """Compute the 12 lexical features for one URL. Never raises."""
    url = strip_www(str(url))
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
        scheme = parsed.scheme.lower()
    except ValueError:  # e.g. broken IPv6 brackets
        host, path, scheme = "", "", ""

    return {
        "url_length": len(url),
        "domain_length": len(host),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_subdomains": count_subdomains(host),
        "uses_https": int(scheme == "https"),
        "has_ip_address": int(is_ip_host(host)),
        "has_at_symbol": int("@" in url),
        # "special" = anything that is not a letter, digit, or one of . - / :
        # (dots and hyphens are counted separately above)
        "num_special_chars": sum(1 for ch in url if not ch.isalnum() and ch not in ".-/:"),
        "suspicious_keyword_count": len(find_suspicious_keywords(url)),
        "path_length": len(path),
        "num_digits": sum(ch.isdigit() for ch in url),
    }


def host_text(url: str) -> str:
    """Lower-cased host without scheme, port, 'user@' part or a leading 'www.' - the text the
    character n-gram model reads, e.g. 'https://www.Secure-Login.example.com/x' -> 'secure-login.example.com'."""
    try:
        host = (urlparse(url.strip()).hostname or "").lower()
    except ValueError:
        return ""
    return re.sub(r"^www\d*\.", "", host)


def normalize_scheme(url: str) -> str:
    """Treat http:// like https:// before scoring, so the scheme (and its 1-character length
    difference) cannot influence the model. HTTPS is reported separately as an indicator."""
    return re.sub(r"^http://", "https://", url.strip(), flags=re.IGNORECASE)


def extract_feature_frame(urls: Iterable[str]) -> pd.DataFrame:
    """Feature matrix (one row per URL) with columns in FEATURE_NAMES order."""
    return pd.DataFrame([extract_features(u) for u in urls], columns=FEATURE_NAMES)
