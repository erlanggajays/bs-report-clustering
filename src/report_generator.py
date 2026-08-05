"""Interactive standalone HTML report generator.

Combines Plotly figures with a Jinja2 template into a single self-contained
`browserstack_report.html`. Plotly's JS is embedded inline (not via CDN), so the
report opens with no network access.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.offline import get_plotlyjs

from exec_metrics import ExecMetrics
from taxonomy import category_description
from triage_engine import FailureCluster

_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"
_DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "output" / "browserstack_report.html"

# Shared Plotly layout so every chart matches the report's visual language.
_LAYOUT = dict(
    margin=dict(l=40, r=20, t=50, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color="#1f2933"),
    colorway=["#2563eb", "#7c3aed", "#db2777", "#ea580c", "#0891b2", "#16a34a"],
)


def _fig_to_div(fig: go.Figure, div_id: str, **layout_overrides) -> str:
    """Render a figure as an embeddable div; JS is injected once, globally.

    ``layout_overrides`` are applied *after* the shared layout, so a chart can
    tweak margins/legend without every other chart inheriting it.
    """
    fig.update_layout(**_LAYOUT)
    if layout_overrides:
        fig.update_layout(**layout_overrides)
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )


def _trim(text: str, limit: int = 58) -> str:
    """Trim a label for axis display, keeping it readable (full text is on hover)."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cluster_chart(clusters: list[FailureCluster]) -> str:
    if not clusters:
        return "<p class='empty'>No failures to cluster — suite is green. 🎉</p>"
    data = pd.DataFrame(
        {
            "Root cause": [f"#{c.cluster_id}: {_trim(c.label)}" for c in clusters],
            "Full": [f"#{c.cluster_id}: {c.label}" for c in clusters],
            "Failures": [c.size for c in clusters],
            "Confidence": [c.confidence for c in clusters],
        }
    )
    fig = px.bar(
        data,
        x="Failures",
        y="Root cause",
        orientation="h",
        color="Confidence",
        color_continuous_scale="Blues",
        range_color=(0, 1),
        custom_data=["Full"],
        title="Failure clusters by size (color = clustering confidence)",
    )
    # Full label on hover; automargin lets the y-axis expand to fit long labels.
    fig.update_traces(hovertemplate="%{customdata[0]}<br>Failures: %{x}<extra></extra>")
    fig.update_layout(
        yaxis=dict(autorange="reversed", automargin=True),
        coloraxis_colorbar=dict(title="conf"),
    )
    return _fig_to_div(fig, "cluster-chart")


def _category_chart(categories: pd.DataFrame | None) -> str:
    if categories is None or categories.empty:
        return ""
    agg = categories.groupby("category", as_index=False)["count"].sum()
    owner_by_cat = dict(zip(categories["category"], categories["owner"]))
    agg["owner"] = agg["category"].map(owner_by_cat)
    agg = agg.sort_values("count")
    fig = px.bar(
        agg,
        x="count",
        y="category",
        orientation="h",
        color="owner",
        title="Failures by category",
    )
    # Legend below the plot (was colliding with the title at the top).
    return _fig_to_div(
        fig,
        "category-chart",
        margin=dict(l=40, r=20, t=44, b=96),
        yaxis=dict(automargin=True, title=None),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, title=None),
    )


def _category_details(classified: pd.DataFrame | None) -> list[dict]:
    """Per-category test breakdown for the expandable table: which tests fell into
    each category, how often, and a link to one example session."""
    if classified is None or classified.empty:
        return []
    out: list[dict] = []
    for cat in classified["category"].value_counts().index:  # busiest first
        sub = classified[classified["category"] == cat]
        has_url = "session_url" in sub.columns
        tests = []
        for name, g in sub.groupby("name"):
            url = ""
            if has_url:
                url = next((u for u in g["session_url"].tolist() if u), "")
            tests.append({
                "name": name or "(unnamed test)",
                "count": int(len(g)),
                "device": g["device"].iloc[0] if "device" in g else "",
                "session_url": url,
            })
        tests.sort(key=lambda t: t["count"], reverse=True)
        out.append({
            "category": cat,
            "owner": sub["owner"].iloc[0],
            "count": int(len(sub)),
            "description": category_description(cat),
            "tests": tests,
        })
    return out


def _device_heatmap(device_risk: pd.DataFrame) -> str:
    if device_risk.empty:
        return "<p class='empty'>No device data available.</p>"
    pivot = device_risk.pivot_table(
        index="device", columns="os_version", values="failure_rate", aggfunc="mean"
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"Android {c}" for c in pivot.columns],
            y=pivot.index,
            colorscale="Reds",
            zmin=0,
            zmax=1,
            colorbar=dict(title="fail rate"),
            hovertemplate="%{y} · %{x}<br>failure rate: %{z:.0%}<extra></extra>",
        )
    )
    fig.update_layout(title="Device / OS failure-rate heatmap")
    return _fig_to_div(fig, "device-heatmap")


def _trend_chart(trend: pd.DataFrame | None) -> str:
    if trend is None or trend.empty or len(trend) < 2:
        return ""  # need at least two builds to show a trend
    pct = (trend["pass_rate"] * 100).round(1)
    labels = list(range(1, len(trend) + 1))
    fig = go.Figure(
        data=go.Scatter(
            x=labels,
            y=pct,
            mode="lines+markers",
            line=dict(color="#2563eb", width=2),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.08)",
            hovertemplate="build %{x}<br>pass rate: %{y}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Suite pass-rate trend (oldest → newest build)",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        xaxis=dict(title="build (chronological)", dtick=1),
    )
    return _fig_to_div(fig, "trend-chart")


def _status_donut(metrics: ExecMetrics) -> str:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Passed", "Failed"],
                values=[metrics.passed, metrics.failed],
                hole=0.62,
                marker=dict(colors=["#16a34a", "#dc2626"]),
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(title="Pass / fail split", showlegend=False)
    return _fig_to_div(fig, "status-donut")


def generate_report(
    build_meta: dict[str, Any],
    metrics: ExecMetrics,
    clusters: list[FailureCluster],
    device_risk: pd.DataFrame,
    flakiness: pd.DataFrame,
    output_path: str | Path | None = None,
    is_sample: bool = False,
    trend: pd.DataFrame | None = None,
    categories: pd.DataFrame | None = None,
    classified: pd.DataFrame | None = None,
) -> Path:
    """Render every artifact into a single standalone HTML file and return its path.

    ``is_sample`` renders a prominent banner marking the report as synthetic mock
    data, so a local test run is never mistaken for a live BrowserStack pull.
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report_template.html")

    flaky_rows = (
        flakiness[flakiness["flakiness_index"] > 0]
        .head(8)
        .to_dict(orient="records")
    )
    # _flakiness_from_history emits a "builds" column; the in-build proxy does not.
    flaky_source = "history" if "builds" in flakiness.columns else "in-build"

    category_rows = (
        categories.sort_values("count", ascending=False).to_dict(orient="records")
        if categories is not None and not categories.empty
        else []
    )

    html = template.render(
        plotly_js=get_plotlyjs(),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        is_sample=is_sample,
        build=build_meta,
        cards=metrics.summary_cards(),
        top_risk_cell=metrics.top_risk_cell,
        clusters=[c.as_dict() for c in clusters],
        flaky_rows=flaky_rows,
        flaky_source=flaky_source,
        cluster_chart=_cluster_chart(clusters),
        device_heatmap=_device_heatmap(device_risk),
        status_donut=_status_donut(metrics),
        trend_chart=_trend_chart(trend),
        category_chart=_category_chart(categories),
        category_rows=category_rows,
        category_details=_category_details(classified),
    )

    out = Path(output_path or _DEFAULT_OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    from ingestor import ingest
    from triage_engine import run_triage
    from exec_metrics import compute_exec_metrics

    frame, meta = ingest(source="file")
    triage = run_triage(frame)
    m = compute_exec_metrics(frame, triage["clusters"], triage["device_anomaly"])
    path = generate_report(
        meta, m, triage["clusters"], triage["device_anomaly"], triage["flakiness"],
        is_sample=True,
    )
    print(f"Report written to {path}")
