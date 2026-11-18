"""Plotly charts styled to match the ThreatLens dark theme."""

import pandas as pd
import plotly.graph_objects as go

COLORS = {
    "bg": "#0A101E",
    "surface": "#111A2E",
    "line": "#22304D",
    "text": "#E7ECF6",
    "muted": "#8E9BB7",
    "accent": "#5B8DEF",
    "ok": "#2FD08F",
    "bad": "#FF5A6E",
    "warn": "#FFB84D",
}
FONT = "IBM Plex Sans, Segoe UI, sans-serif"


def style_fig(fig: go.Figure, height: int = 300, legend: bool = True) -> go.Figure:
    """Apply the shared look: transparent background, muted grid, compact margins."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=COLORS["muted"], size=12),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=12)),
        hoverlabel=dict(bgcolor=COLORS["surface"], font_color=COLORS["text"], bordercolor=COLORS["line"]),
    )
    fig.update_xaxes(gridcolor=COLORS["line"], zerolinecolor=COLORS["line"], linecolor=COLORS["line"])
    fig.update_yaxes(gridcolor=COLORS["line"], zerolinecolor=COLORS["line"], linecolor=COLORS["line"])
    return fig


def verdict_donut(phishing: int, legitimate: int) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=["Phishing", "Legitimate"],
            values=[phishing, legitimate],
            hole=0.7,
            marker=dict(colors=[COLORS["bad"], COLORS["ok"]], line=dict(color=COLORS["surface"], width=3)),
            textinfo="none",
            sort=False,
        )
    )
    total = phishing + legitimate
    fig.add_annotation(text=f"<b>{total}</b><br><span style='font-size:12px'>scans</span>", showarrow=False,
                       font=dict(size=26, color=COLORS["text"], family="Sora, sans-serif"))
    return style_fig(fig, 260)


def risk_bar(counts: dict) -> go.Figure:
    levels = ["Low", "Medium", "High"]
    colors = [COLORS["ok"], COLORS["warn"], COLORS["bad"]]
    fig = go.Figure(go.Bar(x=levels, y=[counts.get(l, 0) for l in levels], marker_color=colors,
                           text=[counts.get(l, 0) for l in levels], textposition="outside",
                           textfont=dict(color=COLORS["text"])))
    fig.update_yaxes(showgrid=True, rangemode="tozero", dtick=1 if max(counts.values(), default=0) < 6 else None)
    return style_fig(fig, 260, legend=False)


def timeline_bar(history: pd.DataFrame) -> go.Figure:
    """Scans per day, stacked by verdict."""
    df = history.copy()
    df["day"] = pd.to_datetime(df["scanned_at"]).dt.date
    pivot = df.groupby(["day", "prediction"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for label, color in (("Legitimate", COLORS["ok"]), ("Phishing", COLORS["bad"])):
        if label in pivot:
            fig.add_bar(x=pivot.index.astype(str), y=pivot[label], name=label, marker_color=color)
    fig.update_layout(barmode="stack", bargap=0.5)
    fig.update_xaxes(type="category")
    fig.update_yaxes(rangemode="tozero")
    return style_fig(fig, 260)


def gauge(p_phishing: float) -> go.Figure:
    """Semicircle gauge of the phishing probability."""
    pct = p_phishing * 100
    color = COLORS["ok"] if pct < 35 else COLORS["bad"] if pct >= 65 else COLORS["warn"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number=dict(suffix="%", font=dict(size=28, color=COLORS["text"], family="Sora, sans-serif"), valueformat=".1f"),
            gauge=dict(
                axis=dict(range=[0, 100], tickvals=[0, 35, 65, 100], tickcolor=COLORS["muted"], tickfont=dict(size=11)),
                bar=dict(color=color, thickness=0.28),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 35], color="rgba(47,208,143,0.16)"),
                    dict(range=[35, 65], color="rgba(255,184,77,0.16)"),
                    dict(range=[65, 100], color="rgba(255,90,110,0.16)"),
                ],
            ),
        )
    )
    fig.update_layout(height=210, margin=dict(l=30, r=30, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, color=COLORS["muted"]))
    return fig


def model_comparison_bar(models: dict) -> go.Figure:
    metric_names = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]
    palette = [COLORS["accent"], COLORS["warn"], COLORS["ok"], COLORS["bad"]]
    fig = go.Figure()
    for (key, label), color in zip(metric_names, palette[:4]):
        fig.add_bar(name=label, x=list(models), y=[models[m][key] for m in models], marker_color=color,
                    text=[f"{models[m][key]:.3f}" for m in models], textposition="outside",
                    textfont=dict(size=11, color=COLORS["text"]))
    fig.update_layout(barmode="group", bargap=0.35)
    fig.update_yaxes(range=[0, 1.1], tickformat=".0%")
    return style_fig(fig, 340)


def confusion_heatmap(cm: dict, title: str) -> go.Figure:
    """cm = {'tn','fp','fn','tp'}; rows = actual, columns = predicted."""
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    text = [[f"<b>{z[r][c]}</b><br><span style='font-size:11px'>{labels[r][c]}</span>" for c in range(2)] for r in range(2)]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=["Legitimate", "Phishing"], y=["Legitimate", "Phishing"], text=text, texttemplate="%{text}",
            colorscale=[[0, "#111A2E"], [1, "#3A63C0"]], showscale=False, hoverinfo="skip",
            xgap=4, ygap=4, textfont=dict(color=COLORS["text"], size=15),
        )
    )
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=COLORS["text"])))
    fig.update_xaxes(title_text="Predicted", side="bottom", showgrid=False)
    fig.update_yaxes(title_text="Actual", autorange="reversed", showgrid=False)
    return style_fig(fig, 320, legend=False)


def roc_chart(models: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                    line=dict(color=COLORS["line"], dash="dash"))
    for (name, m), color in zip(models.items(), (COLORS["accent"], COLORS["ok"])):
        auc = f" (AUC {m['roc_auc']:.3f})" if m.get("roc_auc") is not None else ""
        fig.add_scatter(x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"], mode="lines", name=name + auc,
                        line=dict(color=color, width=3))
    fig.update_xaxes(title_text="False positive rate", range=[0, 1])
    fig.update_yaxes(title_text="True positive rate (recall)", range=[0, 1.02])
    return style_fig(fig, 320)


def importance_bar(importance: dict, height: int = 340) -> go.Figure:
    items = sorted(importance.items(), key=lambda kv: kv[1])
    from .feature_extraction import FEATURE_LABELS  # local import keeps this module light

    fig = go.Figure(
        go.Bar(
            x=[v for _, v in items], y=[FEATURE_LABELS.get(k, k) for k, _ in items], orientation="h",
            marker_color=COLORS["accent"], text=[f"{v * 100:.1f}%" for _, v in items], textposition="outside",
            textfont=dict(size=11, color=COLORS["text"]), cliponaxis=False,
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(showgrid=False)
    fig = style_fig(fig, height, legend=False)
    fig.update_layout(margin=dict(l=8, r=52, t=8, b=8))
    return fig
