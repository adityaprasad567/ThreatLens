"""ThreatLens look and feel: global CSS plus small HTML building blocks."""

import html as htmllib
import re
from urllib.parse import urlparse

import streamlit as st

from .feature_extraction import SUSPICIOUS_KEYWORDS, is_ip_host, split_host

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg:#0A101E; --surface:#111A2E; --surface2:#16213A; --line:#22304D;
  --text:#E7ECF6; --muted:#8E9BB7; --accent:#5B8DEF;
  --ok:#2FD08F; --bad:#FF5A6E; --warn:#FFB84D;
}
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea{
  font-family:'IBM Plex Sans','Segoe UI',sans-serif;
}
.stApp{ background: radial-gradient(1100px 520px at 88% -8%, #13213F 0%, rgba(10,16,30,0) 62%), var(--bg); color:var(--text); }
h1,h2,h3,h4{ font-family:'Sora','Segoe UI',sans-serif !important; letter-spacing:-0.015em; color:var(--text); }
[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer, [data-testid="stAppDeployButton"], .stDeployButton{ visibility:hidden; display:none; }
.block-container{ padding-top:2rem; padding-bottom:4rem; max-width:1240px; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{ background:#0C1325; border-right:1px solid var(--line); }
[data-testid="stSidebarNavLink"]{ border-radius:10px; padding:.55rem .8rem; color:var(--muted); font-weight:500; }
[data-testid="stSidebarNavLink"]:hover{ background:#131E38; color:var(--text); }
[data-testid="stSidebarNavLink"][aria-current="page"]{ background:#172750; color:#fff; box-shadow:inset 3px 0 0 var(--accent); }

/* ---------- inputs & buttons ---------- */
.stTextInput input{
  background:var(--surface2); border:1px solid var(--line); color:var(--text);
  font-family:'JetBrains Mono',monospace; font-size:.95rem; border-radius:10px; padding:.85rem 1rem;
}
.stTextInput input:focus{ border-color:var(--accent); box-shadow:0 0 0 3px rgba(91,141,239,.22); }
[data-testid="stForm"]{ background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:1.3rem 1.4rem; }
.stButton > button, [data-testid="stFormSubmitButton"] button{
  border-radius:10px; border:1px solid var(--line); background:var(--surface2); color:var(--text);
  font-weight:600; padding:.55rem 1.1rem; transition:border-color .15s, background .15s;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover{ border-color:var(--accent); color:#fff; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"], [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primaryFormSubmit"]{
  background:var(--accent); border-color:var(--accent); color:#06122B;
}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover{ background:#7AA3F5; border-color:#7AA3F5; color:#06122B; }
[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:12px; overflow:hidden; }
[data-testid="stAlert"]{ border-radius:12px; }
:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; }

/* ---------- page header ---------- */
.tl-head{ margin-bottom:1.6rem; }
.tl-head h1{ font-size:2rem; margin:0 0 .3rem 0; font-weight:700; }
.tl-head p{ color:var(--muted); margin:0; font-size:1.02rem; max-width:70ch; }

/* ---------- cards ---------- */
.tl-card{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.3rem; height:100%; }
.tl-card h4{ margin:0 0 .5rem 0; font-size:1rem; }
.tl-card p{ color:var(--muted); margin:.2rem 0; font-size:.93rem; line-height:1.55; }
.kpi{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:1rem 1.2rem; border-left:3px solid var(--accent); }
.kpi.ok{ border-left-color:var(--ok); } .kpi.bad{ border-left-color:var(--bad); } .kpi.warn{ border-left-color:var(--warn); }
.kpi .label{ color:var(--muted); font-size:.84rem; font-weight:500; }
.kpi .value{ font-family:'Sora',sans-serif; font-size:2rem; font-weight:700; line-height:1.25; color:var(--text); }
.kpi .note{ color:var(--muted); font-size:.8rem; }
.section-title{ font-family:'Sora',sans-serif; font-size:1.05rem; font-weight:600; margin:1.6rem 0 .7rem 0; color:var(--text); }

/* ---------- chips ---------- */
.chip{ display:inline-flex; align-items:center; gap:.35rem; padding:.2rem .65rem; border-radius:999px; font-size:.78rem; font-weight:600; border:1px solid; margin-right:.4rem; }
.chip.ok{ color:var(--ok); border-color:rgba(47,208,143,.4); background:rgba(47,208,143,.08); }
.chip.bad{ color:var(--bad); border-color:rgba(255,90,110,.4); background:rgba(255,90,110,.08); }
.chip.warn{ color:var(--warn); border-color:rgba(255,184,77,.4); background:rgba(255,184,77,.08); }
.chip.info{ color:#9CB4FF; border-color:rgba(91,141,239,.4); background:rgba(91,141,239,.08); }

/* ---------- verdict ---------- */
.verdict{ border-radius:16px; padding:1.4rem 1.6rem; border:1px solid; }
.verdict.bad{ border-color:rgba(255,90,110,.5); background:rgba(255,90,110,.07); }
.verdict.ok{ border-color:rgba(47,208,143,.5); background:rgba(47,208,143,.07); }
.verdict.warn{ border-color:rgba(255,184,77,.5); background:rgba(255,184,77,.07); }
.verdict .title{ font-family:'Sora',sans-serif; font-size:1.7rem; font-weight:700; margin:.2rem 0 .5rem 0; }
.verdict.bad .title{ color:var(--bad); } .verdict.ok .title{ color:var(--ok); } .verdict.warn .title{ color:var(--warn); }
.verdict .sub{ color:var(--muted); font-size:.93rem; line-height:1.55; }

/* ---------- URL anatomy ---------- */
.anatomy{ font-family:'JetBrains Mono',monospace; font-size:1rem; line-height:2.1; word-break:break-all;
  background:#0B1324; border:1px solid var(--line); border-radius:12px; padding:1rem 1.2rem; }
.anatomy .seg{ padding:2px 3px; border-radius:5px; }
.seg.ok{ color:var(--ok); } .seg.warn{ color:var(--warn); } .seg.sub{ color:#9CB4FF; } .seg.dom{ color:#fff; font-weight:500; }
.seg.port{ color:#9CB4FF; } .seg.path{ color:var(--muted); } .seg.bad{ background:rgba(255,90,110,.2); color:#FF8D9B; }
.anatomy .punc{ color:#51607F; }
.anatomy mark{ background:rgba(255,184,77,.22); color:#FFD08A; border-radius:4px; padding:0 2px; }
.legend{ display:flex; flex-wrap:wrap; gap:.4rem 1.1rem; margin-top:.6rem; font-size:.78rem; color:var(--muted); }
.legend i{ display:inline-block; width:.6rem; height:.6rem; border-radius:3px; margin-right:.35rem; }

/* ---------- indicators ---------- */
.ind{ display:flex; gap:.85rem; padding:.85rem 1rem; border-radius:12px; border:1px solid var(--line); background:var(--surface); margin-bottom:.55rem; border-left-width:3px; }
.ind.high{ border-left-color:var(--bad); } .ind.medium{ border-left-color:var(--warn); } .ind.low{ border-left-color:var(--accent); } .ind.good{ border-left-color:var(--ok); }
.ind svg{ flex:none; margin-top:2px; }
.ind .t{ font-weight:600; font-size:.95rem; color:var(--text); }
.ind .d{ color:var(--muted); font-size:.87rem; line-height:1.5; margin-top:.15rem; }

/* ---------- tables & callouts ---------- */
.tl-table th, .tl-table td{ border:none; border-bottom:1px solid var(--line); }
.tl-table td{ vertical-align:middle; }
.tl-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:.9rem; }
.kpi .value.sm{ font-size:1.35rem; padding:.3rem 0; }
.tl-table{ width:100%; border-collapse:collapse; font-size:.9rem; }
.tl-table th{ text-align:left; color:var(--muted); font-weight:500; padding:.55rem .7rem; border-bottom:1px solid var(--line); }
.tl-table td{ padding:.55rem .7rem; border-bottom:1px solid rgba(34,48,77,.6); color:var(--text); }
.tl-table td.num{ font-family:'JetBrains Mono',monospace; }
.callout{ border-radius:12px; padding:.85rem 1.1rem; border:1px solid; font-size:.9rem; line-height:1.55; margin:.4rem 0 1rem 0; }
.callout.warn{ border-color:rgba(255,184,77,.45); background:rgba(255,184,77,.07); color:#FFDDA6; }
.callout.info{ border-color:rgba(91,141,239,.4); background:rgba(91,141,239,.07); color:#C3D3FA; }
.side-status{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:.8rem .95rem; font-size:.82rem; color:var(--muted); line-height:1.5; }
.side-status b{ color:var(--text); }
.dot{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.4rem; }
.dot.ok{ background:var(--ok); } .dot.warn{ background:var(--warn); }

@media (max-width:760px){
  .tl-head h1{ font-size:1.5rem; } .verdict .title{ font-size:1.35rem; } .kpi .value{ font-size:1.6rem; }
  .anatomy{ font-size:.86rem; }
}
@media (prefers-reduced-motion:reduce){ *{ transition:none !important; animation:none !important; } }
"""

# Feather-style 24x24 stroke icons used inside HTML blocks.
ICONS = {
    "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="M22 4 12 14.01l-3-3"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
}
SEVERITY_STYLE = {"high": ("alert", "#FF5A6E"), "medium": ("alert", "#FFB84D"), "low": ("info", "#5B8DEF"), "good": ("check", "#2FD08F")}
RISK_TONE = {"Low": "ok", "Medium": "warn", "High": "bad"}


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render HTML. Lines are joined so Markdown never mistakes indentation for a code block."""
    st.markdown(" ".join(line.strip() for line in markup.strip().splitlines()), unsafe_allow_html=True)


def esc(text) -> str:
    return htmllib.escape(str(text), quote=True)


def icon_svg(name: str, color: str, size: int = 20) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


# ------------------------------------------------------------ components
def page_header(title: str, subtitle: str) -> None:
    html(f'<div class="tl-head"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>')


def section_title(text: str) -> None:
    html(f'<div class="section-title">{esc(text)}</div>')


def kpi(label: str, value, note: str = "", tone: str = "") -> str:
    size = " sm" if len(str(value)) > 10 else ""  # long values (dates, model names) get a smaller font
    return (f'<div class="kpi {tone}"><div class="label">{esc(label)}</div>'
            f'<div class="value{size}">{esc(value)}</div><div class="note">{esc(note)}</div></div>')


def chip(text: str, tone: str = "info") -> str:
    return f'<span class="chip {tone}">{esc(text)}</span>'


def callout(text: str, tone: str = "info") -> None:
    html(f'<div class="callout {tone}">{esc(text)}</div>')


def sample_data_warning() -> None:
    callout("These numbers come from the small synthetic SAMPLE dataset, so they only prove the pipeline runs. "
            "They are not real model performance. Train on the PhiUSIIL dataset (see README) for meaningful results.", "warn")


def domain_mode_note(result: dict) -> None:
    """Explain that only the domain influenced the score (domain-mode models)."""
    if result.get("training_mode") == "domain":
        callout(f"This model scores the domain only ({result['scored_url']}). The path and parameters are checked by the "
                "rule-based indicators below, but they do not change the score.", "info")


def verdict_card(result: dict) -> None:
    """Large result card: verdict, confidence and risk chips, short explanation."""
    is_phish = result["prediction"] == "Phishing"
    review = result.get("manual_review", False)
    tone = "bad" if is_phish else ("warn" if review or result["risk_level"] == "Medium" else "ok")
    title = "Flagged as likely phishing" if is_phish else ("Model says legitimate, but check this link" if review else "Looks legitimate")
    sv = result.get("site_verdict")
    if sv:
        html(f"""
    <div class="verdict {sv['tone']}">
      {chip(sv['label'], sv['tone'])}
      <div class="title">{esc(sv['headline'])}</div>
      <div class="sub">{esc(sv['reason'])}</div>
    </div>
    """)
    html(f"""
    <div class="verdict {tone}">
      {chip("Confidence " + f"{result['confidence'] * 100:.1f}%", "info")}
      {chip("Risk: " + result["risk_level"], RISK_TONE[result["risk_level"]])}
      <div class="title">{title}</div>
      <div class="sub">{esc(result["explanation"])}</div>
    </div>
    """)


def indicator_rows(indicators: list[dict], positives: list[str] | None = None) -> None:
    for item in indicators:
        name, color = SEVERITY_STYLE[item["severity"]]
        html(f'<div class="ind {item["severity"]}">{icon_svg(name, color)}<div><div class="t">{esc(item["title"])}</div>'
             f'<div class="d">{esc(item["detail"])}</div></div></div>')
    for text in positives or []:
        name, color = SEVERITY_STYLE["good"]
        html(f'<div class="ind good">{icon_svg(name, color)}<div><div class="t">{esc(text)}</div></div></div>')


def _mark_keywords(text: str, enabled: bool = True) -> str:
    """Escape text and wrap suspicious keywords in <mark>."""
    if not enabled:
        return esc(text)
    pattern = re.compile("(" + "|".join(SUSPICIOUS_KEYWORDS) + ")", re.IGNORECASE)
    parts = pattern.split(text)
    return "".join(f"<mark>{esc(p)}</mark>" if i % 2 else esc(p) for i, p in enumerate(parts))


def url_anatomy(url: str, mark_keywords: bool = True, show_legend: bool = True) -> None:
    """Show the URL split into protocol / subdomain / domain / path with red flags highlighted."""
    try:
        parsed = urlparse(url)
    except ValueError:
        html(f'<div class="anatomy">{esc(url)}</div>')
        return

    scheme, netloc = parsed.scheme, parsed.netloc
    userinfo, hostport = ("", netloc)
    if "@" in netloc:
        userinfo, hostport = netloc.rsplit("@", 1)
    match = re.match(r"^(.*?)(:\d+)?$", hostport)
    host, port = (match.group(1), match.group(2) or "") if match else (hostport, "")

    start = url.find("://") + 3
    rest = url[start + len(netloc):]

    parts = [f'<span class="seg {"ok" if scheme == "https" else "warn"}">{esc(scheme)}</span><span class="punc">://</span>']
    if userinfo:
        parts.append(f'<span class="seg bad">{esc(userinfo)}@</span>')
    if is_ip_host(host.strip("[]").lower()):
        parts.append(f'<span class="seg bad">{esc(host)}</span>')
    else:
        sub, _ = split_host(host)
        if sub:
            parts.append(f'<span class="seg sub">{_mark_keywords(host[:len(sub)], mark_keywords)}</span><span class="punc">.</span>')
            parts.append(f'<span class="seg dom">{_mark_keywords(host[len(sub) + 1:], mark_keywords)}</span>')
        else:
            parts.append(f'<span class="seg dom">{_mark_keywords(host, mark_keywords)}</span>')
    if port:
        parts.append(f'<span class="seg port">{esc(port)}</span>')
    if rest:
        parts.append(f'<span class="seg path">{_mark_keywords(rest, mark_keywords)}</span>')

    legend = (
        '<div class="legend">'
        '<span><i style="background:#2FD08F"></i>HTTPS</span><span><i style="background:#FFB84D"></i>HTTP / keyword</span>'
        '<span><i style="background:#9CB4FF"></i>Subdomain</span><span><i style="background:#fff"></i>Registered domain</span>'
        '<span><i style="background:#8E9BB7"></i>Path and query</span><span><i style="background:#FF5A6E"></i>Red flag (IP or @)</span>'
        '</div>'
    )
    html(f'<div class="anatomy">{"".join(parts)}</div>{legend if show_legend else ""}')
