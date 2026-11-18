"""Popular-domain list (top ~100k) used to (1) reduce false positives and (2) add
varied legitimate examples to training. Shared-hosting domains are excluded because
anyone can create a page on them (e.g. evil.blogspot.com)."""

from functools import lru_cache

from .config import DATA_DIR
from .feature_extraction import is_ip_host, split_host

TOP_DOMAINS_PATH = DATA_DIR / "top_domains.txt"

RESERVED_OR_TEST_DOMAINS = {
    "example.com", "example.org", "example.net", "invalid", "localhost", "test",
}

SHARED_HOSTING = {
    "blogspot.com", "blogspot.in", "blogspot.ru", "weebly.com", "github.io", "gitlab.io",
    "herokuapp.com", "netlify.app", "wixsite.com", "wix.com", "web.app", "firebaseapp.com",
    "pages.dev", "workers.dev", "000webhostapp.com", "wordpress.com", "tumblr.com",
    "sites.google.com", "appspot.com", "azurewebsites.net", "cloudfront.net", "amazonaws.com",
    "vercel.app", "glitch.me", "repl.co", "godaddysites.com", "webflow.io", "square.site",
    "myshopify.com", "bravenet.com", "yolasite.com", "jimdofree.com", "ukit.me", "page.link",
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "rebrand.ly", "cutt.ly",
    "forms.gle", "typeform.com", "notion.site", "carrd.co", "weeblysite.com", "mystrikingly.com",
    "dropbox.com", "drive.google.com", "docs.google.com", "storage.googleapis.com",
}


@lru_cache(maxsize=1)
def load_top_domains() -> frozenset:
    if not TOP_DOMAINS_PATH.exists():
        return frozenset()
    names = TOP_DOMAINS_PATH.read_text(encoding="utf-8").split()
    return frozenset(n for n in names if n not in SHARED_HOSTING and n not in RESERVED_OR_TEST_DOMAINS)


def registered_domain(host: str) -> str:
    return split_host(host)[1]


def is_trusted_host(host: str) -> bool:
    host = (host or "").lower().strip(".")
    if not host or is_ip_host(host):
        return False
    domain = registered_domain(host)
    # A popular registrable domain does not make every subdomain trustworthy.
    # Only trust the apex and its conventional www host; user-created subdomains
    # must be assessed independently.
    return domain in load_top_domains() and host in {domain, f"www.{domain}"}
