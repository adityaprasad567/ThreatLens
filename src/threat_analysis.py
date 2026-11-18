"""Turn raw URL features into human-readable suspicious indicators and explanations.

These URL rules complement the trained model. Narrow structural-deception rules
can override a reassuring model score; generic keywords alone never prove phishing.
"""

from urllib.parse import urlparse

from .feature_extraction import find_suspicious_keywords, split_host

DISCLAIMER = (
    "This is a machine-learning estimate based only on the text of the URL. "
    "It is not verified threat intelligence, and a low score does not prove a site is safe."
)


def get_indicators(url: str, features: dict) -> list[dict]:
    """Return suspicious indicators, most severe first.

    Each item: {"title", "detail", "severity": "high" | "medium" | "low"}
    """
    found = []

    def add(title: str, detail: str, severity: str) -> None:
        found.append({"title": title, "detail": detail, "severity": severity})

    if features["has_ip_address"]:
        add("IP address instead of a domain name",
            "Legitimate services almost always use a registered domain name. Raw IP addresses are common in throw-away phishing pages.",
            "high")
    if features["has_at_symbol"]:
        add("Contains an @ symbol",
            "Browsers ignore everything before @ in the address. Attackers use it to make a link look like a trusted site.",
            "high")
    keywords = find_suspicious_keywords(url)
    # Brand names in a subdomain or look-alike registered domain are not proof
    # on their own, but are a strong impersonation signal when the registrable
    # domain is not the brand's known domain.
    known_brands = {
        "amazon": {"amazon.com", "amazon.co.uk", "amazon.in", "amazon.de", "amazon.ca", "amazon.co.jp"},
        "paypal": {"paypal.com"}, "google": {"google.com", "google.co.uk", "google.co.in"},
        "microsoft": {"microsoft.com"}, "apple": {"apple.com"}, "facebook": {"facebook.com"},
        "instagram": {"instagram.com"}, "netflix": {"netflix.com"}, "linkedin": {"linkedin.com"},
        "github": {"github.com"}, "whatsapp": {"whatsapp.com"}, "outlook": {"outlook.com"},
    }
    try:
        host = (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        host = ""
    subdomain, registered = split_host(host) if host else ("", "")
    host_labels = set(host.split("."))
    for brand, allowed_domains in known_brands.items():
        brand_in_host = brand in host_labels or any(brand in label for label in host.split("."))
        if brand_in_host and registered not in allowed_domains:
            add("Possible brand impersonation",
                f"The hostname uses '{brand}' but its registered domain is '{registered}', not a known {brand} domain. Check the address through the brand's official app or website.",
                "high")
            break
    if len(keywords) >= 3:
        add(f"{len(keywords)} suspicious keywords",
            "Found: " + ", ".join(keywords) + ". Several credential-related words together are typical of phishing lures.",
            "high")
    elif keywords:
        add(f"Suspicious keyword{'s' if len(keywords) > 1 else ''}: {', '.join(keywords)}",
            "Words like these are used to create urgency or imitate account pages (they also appear on genuine sites).",
            "medium")
    if not features["uses_https"]:
        add("Connection is not encrypted (HTTP)",
            "The page does not use HTTPS. Note that many phishing sites do use HTTPS, so HTTPS alone is not a sign of safety.",
            "medium")
    if features["num_subdomains"] >= 3:
        add(f"{int(features['num_subdomains'])} subdomains",
            "Deeply nested subdomains can hide the real domain, e.g. bank.com.attacker.xyz.",
            "medium")
    if features["num_hyphens"] >= 3:
        add(f"{int(features['num_hyphens'])} hyphens",
            "Many hyphens are common in look-alike domains such as secure-bank-login-update.",
            "medium")
    if features["url_length"] > 75:
        add(f"Long URL ({int(features['url_length'])} characters)",
            "Long URLs can bury the real destination or hide suspicious parts out of sight.",
            "medium")
    if features["domain_length"] > 30:
        add(f"Long domain name ({int(features['domain_length'])} characters)",
            "Unusually long domains are often auto-generated or built to imitate a brand.",
            "medium")
    if features["num_dots"] >= 5:
        add(f"{int(features['num_dots'])} dots",
            "A high number of dots usually means many subdomain levels or a complex path.",
            "low")
    if features["num_digits"] >= 8:
        add(f"{int(features['num_digits'])} digits",
            "Long runs of digits often come from tracking IDs or auto-generated pages.",
            "low")
    if features["num_special_chars"] >= 5:
        add(f"{int(features['num_special_chars'])} special characters",
            "Lots of symbols (?, =, &, %, _) can indicate encoded or obfuscated parameters.",
            "low")
    if features["path_length"] > 60:
        add(f"Long path ({int(features['path_length'])} characters)",
            "Very long paths are sometimes used to disguise the page being requested.",
            "low")

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(found, key=lambda item: order[item["severity"]])


def get_positive_signals(features: dict) -> list[str]:
    """Neutral or reassuring traits. These are context, never a guarantee of safety."""
    signals = []
    if features["uses_https"]:
        signals.append("Uses HTTPS (encrypted connection)")
    if not features["has_ip_address"]:
        signals.append("Uses a domain name rather than a raw IP address")
    if features["suspicious_keyword_count"] == 0:
        signals.append("No suspicious keywords in the URL")
    if features["num_subdomains"] <= 1:
        signals.append("Simple subdomain structure")
    if features["url_length"] <= 75:
        signals.append("Short, readable URL")
    return signals


def build_explanation(prediction: str, p_phishing: float, risk: str, model_name: str, indicators: list[dict],
                      manual_review: bool = False) -> str:
    """One short paragraph explaining the result in plain language."""
    pct = f"{p_phishing * 100:.1f}%"
    top = [i["title"].lower() for i in indicators[:3]]

    if prediction == "Phishing":
        text = f"The {model_name} model estimates a {pct} probability that this URL is phishing ({risk} risk)."
        if top:
            text += " It is especially notable for: " + "; ".join(top) + "."
        else:
            text += " No single rule-of-thumb indicator stands out, so the model is reacting to a combination of features."
    else:
        text = f"The {model_name} model estimates only a {pct} probability of phishing ({risk} risk), so the URL looks more like the legitimate examples it was trained on."
        if top:
            text += " Still worth noticing: " + "; ".join(top) + "."
        if manual_review:
            text += " Because a serious red flag is present, do not treat this as safe: check the link manually before opening it."
        elif risk == "Medium":
            text += " The score is close to the decision boundary, so treat this result with extra caution."
    return text
