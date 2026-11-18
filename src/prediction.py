
"""Load the trained ThreatLens model and score URLs.

The URL is analysed locally. No website is opened.

The saved machine-learning model is preserved.
Rule-based signals are applied separately from the raw
machine-learning probability.
"""

from urllib.parse import urlparse

import joblib
import pandas as pd

from .config import (
    MODEL_PATH,
    RISK_HIGH_MIN,
    RISK_LOW_MAX,
)

from .feature_extraction import (
    extract_features,
    normalize_scheme,
    to_domain_url,
    validate_url,
)

from .trusted import is_trusted_host

from .threat_analysis import (
    DISCLAIMER,
    build_explanation,
    get_indicators,
    get_positive_signals,
)


class ModelNotFoundError(Exception):
    """Raised when the trained model does not exist."""


def load_model() -> dict:
    """Load the existing trained model bundle."""

    if not MODEL_PATH.exists():
        raise ModelNotFoundError(
            "No trained model found. Run "
            "`python -m src.train_model --data "
            "data/Combined_PhiUSIIL_PhiLegit.csv` first."
        )

    return joblib.load(MODEL_PATH)


def risk_level(p_phishing: float) -> str:
    """Return Low, Medium, or High risk."""

    if p_phishing >= RISK_HIGH_MIN:
        return "High"

    if p_phishing < RISK_LOW_MAX:
        return "Low"

    return "Medium"


def get_hostname(url: str) -> str:
    """Extract the actual hostname safely."""

    try:
        parsed = urlparse(normalize_scheme(url))
        return (parsed.hostname or "").lower().rstrip(".")

    except (ValueError, TypeError):
        return ""


def get_registered_domain(host: str) -> str:
    """Approximate registered domain.

    For more accurate handling of public suffixes,
    use a maintained Public Suffix List library.
    """

    labels = host.lower().rstrip(".").split(".")

    if len(labels) < 2:
        return host

    two_part_suffixes = {
        "co.uk", "org.uk", "ac.uk",
        "co.in", "org.in", "net.in",
        "com.au", "net.au",
        "co.jp", "com.br",
    }

    suffix = ".".join(labels[-2:])

    if suffix in two_part_suffixes and len(labels) >= 3:
        return ".".join(labels[-3:])

    return ".".join(labels[-2:])


def has_deceptive_userinfo(url: str) -> bool:
    """Detect misleading @ user-info in the URL."""

    try:
        parsed = urlparse(normalize_scheme(url))

        return (
            parsed.username is not None
            or parsed.password is not None
        )

    except (ValueError, TypeError):
        return False


def has_brand_impersonation(host: str) -> bool:
    """Detect common brand names used on unrelated domains.

    This is a heuristic and not proof of malicious activity.
    """

    brands = {
        "google": {"google.com", "google.co.in"},
        "github": {"github.com"},
        "paypal": {"paypal.com"},
        "amazon": {"amazon.com", "amazon.in"},
        "microsoft": {"microsoft.com"},
        "facebook": {"facebook.com"},
        "instagram": {"instagram.com"},
        "netflix": {"netflix.com"},
        "apple": {"apple.com"},
    }

    registered = get_registered_domain(host)

    normalized_host = host.lower().replace("-", "")

    for brand, official_domains in brands.items():

        if brand not in normalized_host:
            continue

        if registered in official_domains:
            continue

        return True

    return False


def site_verdict(
    p_phishing: float,
    indicators: list[dict],
    trusted: bool,
) -> dict:
    """Build the user-friendly verdict."""

    high_flag = any(
        item.get("severity") == "high"
        for item in indicators
    )

    if p_phishing >= RISK_HIGH_MIN:
        return {
            "label": "Likely FAKE",
            "tone": "bad",
            "headline": "This website looks fake",
            "reason": (
                "The URL contains patterns associated with "
                "phishing. Avoid entering passwords, OTPs "
                "or payment information."
            ),
        }

    if p_phishing < RISK_LOW_MAX and not high_flag:

        reason = (
            "The hostname matches a recognized domain."
            if trusted
            else
            "The URL has a relatively low estimated risk."
        )

        return {
            "label": "Likely REAL",
            "tone": "ok",
            "headline": "This website looks real",
            "reason": (
                reason
                + " This is not a guarantee that the page is safe."
            ),
        }

    return {
        "label": "NOT SURE",
        "tone": "warn",
        "headline": "Cannot confirm this website is safe",
        "reason": (
            "The URL has mixed or uncertain signals. "
            "Verify the website independently."
        ),
    }


def predict_url(
    url: str,
    bundle: dict | None = None,
) -> dict:
    """Predict phishing risk for a URL."""

    is_valid, cleaned, message = validate_url(url)

    if not is_valid:
        return {
            "ok": False,
            "error": message,
        }

    if bundle is None:
        bundle = load_model()

    model = bundle["model"]

    mode = bundle.get("training_mode", "full")

    # Preserve support for older domain-only models.
    scored_url = (
        to_domain_url(cleaned)
        if mode == "domain"
        else cleaned
    )

    full_features = extract_features(cleaned)

    # Use the same URL normalization expected by
    # the trained model.
    features = extract_features(
        normalize_scheme(scored_url)
    )

    # Preserve the original HTTPS feature.
    features["uses_https"] = full_features["uses_https"]

    row = pd.DataFrame(
        [features],
        columns=bundle["feature_names"],
    )

    if bundle.get("uses_host_text"):
        row["host_text"] = (
            normalize_scheme(scored_url).lower()
        )

    probabilities = model.predict_proba(row)[0]

    phishing_index = list(
        model.classes_
    ).index(1)

    model_probability = float(
        probabilities[phishing_index]
    )

    # -------------------------------------------
    # URL ANALYSIS
    # -------------------------------------------

    indicators = get_indicators(
        cleaned,
        full_features,
    )

    host = get_hostname(cleaned)

    trusted = is_trusted_host(host)

    deceptive_userinfo = has_deceptive_userinfo(
        cleaned
    )

    brand_impersonation = has_brand_impersonation(
        host
    )

    structural_warning = (
        deceptive_userinfo
        or brand_impersonation
    )

    # -------------------------------------------
    # FINAL RISK SCORE
    # -------------------------------------------

    p_phishing = model_probability

    # A trusted host is a supporting signal.
    # It is not an unconditional safety guarantee.
    #
    # Only reduce risk for an exact trusted host
    # without structural deception indicators.
    #
    # This adjustment is heuristic and should
    # be calibrated against a validation dataset.

    if trusted and not structural_warning:

        p_phishing = min(
            p_phishing,
            0.20,
        )

    # Structural deception is stronger than
    # generic keywords such as login or verify.
    #
    # Do not force a 100% probability.

    if structural_warning:

        p_phishing = max(
            p_phishing,
            0.85,
        )

    p_phishing = max(
        0.0,
        min(1.0, p_phishing),
    )

    # -------------------------------------------
    # CLASSIFICATION
    # -------------------------------------------

    prediction = (
        "Phishing"
        if p_phishing >= 0.5
        else "Legitimate"
    )

    confidence = max(
        p_phishing,
        1.0 - p_phishing,
    )

    risk = risk_level(p_phishing)

    high_indicators = any(
        item.get("severity") == "high"
        for item in indicators
    )

    manual_review = (
        prediction == "Legitimate"
        and high_indicators
    )

    rule_override = structural_warning

    verdict = site_verdict(
        p_phishing,
        indicators,
        trusted,
    )

    # Do not reassure the user when a serious
    # indicator conflicts with the score.
    if manual_review:
        verdict = {
            "label": "NOT SURE",
            "tone": "warn",
            "headline": (
                "This URL needs additional verification"
            ),
            "reason": (
                "The model estimates low risk, but "
                "some warning indicators remain. "
                "Verify the destination independently."
            ),
        }

    explanation = build_explanation(
        prediction,
        p_phishing,
        risk,
        bundle["model_name"],
        indicators,
        manual_review,
    )

    if deceptive_userinfo:
        explanation += (
            " The URL contains deceptive user-info "
            "syntax using an @ symbol."
        )

    if brand_impersonation:
        explanation += (
            " The hostname contains a recognizable "
            "brand name but does not match its "
            "expected registered domain."
        )

    if trusted:
        explanation += (
            " The hostname matches the trusted-domain "
            "list. This is supporting evidence only, "
            "not proof of website safety."
        )

    return {
        "ok": True,
        "url": cleaned,
        "prediction": prediction,
        "phishing_probability": p_phishing,
        "confidence": confidence,
        "risk_level": risk,
        "site_verdict": verdict,
        "manual_review": manual_review,
        "training_mode": mode,
        "scored_url": scored_url,
        "features": features,
        "indicators": indicators,
        "positive_signals": get_positive_signals(
            full_features
        ),
        "trusted_domain": trusted,
        "model_probability": model_probability,
        "rule_override": rule_override,
        "detection_method": (
            "hybrid (machine learning + URL safety rules)"
        ),
        "explanation": explanation,
        "model_name": bundle["model_name"],
        "dataset_kind": bundle.get(
            "dataset_kind",
            "unknown",
        ),
        "disclaimer": DISCLAIMER,
    }
