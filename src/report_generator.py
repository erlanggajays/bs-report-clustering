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

from config import settings
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


# Plain-English framing for each dimension: how to name it, and what a finding on
# it should prompt the reader to do. Written for a lead skimming the report, not a
# statistician — the numbers stay, but as supporting detail rather than the headline.
_FINDING_PHRASING = {
    "feature_area": (
        "{level} tests fail {ratio}× more often than the rest of the suite",
        "other areas",
        "Concentrate investigation here — this is where failures cluster.",
    ),
    "device": (
        "Tests on {level} fail {ratio}× more often than on other devices",
        "other devices",
        "Looks environment-specific. Reproduce on this device before treating it as a product bug.",
    ),
    "os_version": (
        "Tests on Android {level} fail {ratio}× more often than on other versions",
        "other OS versions",
        "Looks OS-specific. Check behaviour differences on this version.",
    ),
    "platform": (
        "{level} fails {ratio}× more often than the other platform",
        "the other platform",
        "One platform's implementation is behind — compare the two.",
    ),
    "app_version": (
        "Build {level} fails {ratio}× more often than other builds",
        "other builds",
        "Check what changed in this build.",
    ),
}


def _evidence_strength(p_value: float) -> str:
    """Translate a p-value into words, so the reader need not interpret one."""
    if p_value < 0.001:
        return "very strong evidence"
    if p_value < 0.01:
        return "strong evidence"
    return "moderate evidence"


def _finding_cards(findings: pd.DataFrame | None) -> list[dict]:
    """Turn each statistical finding into a plain-language card.

    Keeps the odds ratio, confidence interval and p-value, but as a muted footnote:
    the headline is a sentence a lead can act on without reading statistics.
    """
    if findings is None or findings.empty:
        return []
    cards = []
    for i, f in enumerate(findings.itertuples(), start=1):
        template, comparison, so_what = _FINDING_PHRASING.get(
            f.dimension,
            ("{level} fails {ratio}× more often than the rest",
             "the rest", "Worth investigating."),
        )
        level = str(f.level)
        # Round for the headline — a lead does not need two decimal places; the exact
        # value stays in the statistics footnote.
        headline = template.format(
            level=level[:1].upper() + level[1:], ratio=f"{round(f.odds_ratio, 1):g}"
        )
        cards.append({
            "rank": i,
            "headline": headline,
            "failures": int(f.failures),
            "sessions": int(f.sessions),
            "rate_pct": round(f.failure_rate * 100),
            "baseline_pct": round(f.baseline_rate * 100),
            "comparison": comparison,
            "so_what": so_what,
            "strength": _evidence_strength(float(f.p_value)),
            "odds_ratio": f.odds_ratio,
            "ci_low": f.ci_low,
            "ci_high": f.ci_high,
            "p_value": float(f.p_value),
            "confounded_with": getattr(f, "confounded_with", "") or "",
            "dimension": f.dimension,
            "level": level,
        })
    return cards


# Owner -> CSS suffix, so the badge colour carries urgency rather than being uniform.
# A product defect reaches users; an infra flake just needs a re-run. Reading the
# panel should convey that ordering before any number is read.
_OWNER_CLASS = {
    "Product bug": "own-product",      # red    — a real defect, highest urgency
    "Backend": "own-backend",          # orange — a real failure, service side
    "Test automation": "own-test",      # blue   — our tooling, not user-facing
    "Test setup": "own-test",
    "Needs triage": "own-triage",      # purple — unknown, needs a human
    "Infra / re-run": "own-infra",     # grey   — lowest, just re-run
}


def owner_class(owner: str) -> str:
    """CSS suffix for an owner badge; unknown owners fall back to the neutral style."""
    return _OWNER_CLASS.get(owner, "own-test")


def _records(frame: pd.DataFrame | None, limit: int | None = None) -> list[dict]:
    """DataFrame -> list of dicts for the template ([] when absent/empty)."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    out = frame.head(limit) if limit else frame
    return out.to_dict(orient="records")


def _platform_names(frame: pd.DataFrame | None) -> list[str]:
    """Platform column names from the area x platform pivot (for table headers)."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    return [c for c in frame.columns if c != "feature_area"]


def _trim(text: str, limit: int = 58) -> str:
    """Trim a label for axis display, keeping it readable (full text is on hover)."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cluster_chart(clusters: list[FailureCluster], top: int = 12) -> str:
    if not clusters:
        return "<p class='empty'>No failures to cluster — suite is green. 🎉</p>"
    # Clusters arrive largest-first. Chart the meaningful ones and roll the long
    # tail of one-off failures into a single bar, so it can't dominate the axis.
    head, tail = clusters[:top], clusters[top:]
    labels = [f"#{c.cluster_id}: {_trim(c.label)}" for c in head]
    rows = [
        {
            "Root cause": label,
            "Full": f"#{c.cluster_id}: {c.label}",
            "Failures": c.size,
            "Confidence": c.confidence,
        }
        for c, label in zip(head, labels)
    ]
    if tail:
        labels.append(f"+ {len(tail)} smaller clusters")
        rows.append({
            "Root cause": labels[-1],
            "Full": f"{len(tail)} clusters of {sum(c.size for c in tail)} failures total",
            "Failures": sum(c.size for c in tail),
            "Confidence": 0.0,
        })
    # Plotly right-aligns categorical y labels, which leaves a ragged left edge on
    # error text of wildly differing lengths. Padding every label to the same
    # character count in a monospace tick font aligns them left instead. A
    # non-breaking space is used because trailing spaces collapse in HTML.
    width = max(len(label) for label in labels)
    padded = {label: label + "\u00a0" * (width - len(label)) for label in labels}
    for row in rows:
        row["Root cause"] = padded[row["Root cause"]]
    data = pd.DataFrame(rows)
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
        yaxis=dict(autorange="reversed", automargin=True, title=None,
                   tickfont=dict(family="ui-monospace, Menlo, monospace", size=11)),
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


def _root_causes_by_category(clusters: list[FailureCluster]) -> dict[str, list[dict]]:
    """Group clusters under their category, largest first.

    A single category routinely contains several genuinely different root causes —
    element-not-found split into seven on real data — so the category alone cannot
    say what to fix first. Nesting keeps that detail without a second table.
    """
    grouped: dict[str, list[dict]] = {}
    for cluster in sorted(clusters, key=lambda c: c.size, reverse=True):
        grouped.setdefault(cluster.category or "uncategorized", []).append({
            "label": cluster.label,
            "size": cluster.size,
            "confidence_pct": round(cluster.confidence * 100),
            "session_url": cluster.session_urls[0] if cluster.session_urls else "",
        })
    return grouped


def _category_details(classified: pd.DataFrame | None,
                      clusters: list[FailureCluster] | None = None) -> list[dict]:
    """Per-category test breakdown for the expandable table: which tests fell into
    each category, how often, and a link to one example session."""
    if classified is None or classified.empty:
        return []
    root_causes = _root_causes_by_category(clusters or [])
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
            "owner_class": owner_class(sub["owner"].iloc[0]),
            "count": int(len(sub)),
            # The header counts failures while the list groups by test, so one test
            # failing twice shows a single row against a count of 2. Carrying the
            # test count lets the report state that relationship instead of leaving
            # the reader to spot a small "x2".
            "test_count": len(tests),
            "description": category_description(cat),
            "root_causes": (root_causes or {}).get(cat, []),
            "tests": tests,
        })
    return out


def _feature_area_chart(fa: pd.DataFrame | None) -> str:
    if fa is None or fa.empty:
        return ""
    d = fa[fa["sessions"] >= 3].sort_values("failure_rate")
    if d.empty:
        return ""
    fig = px.bar(
        d, x="failure_rate", y="feature_area", orientation="h",
        color="failure_rate", color_continuous_scale="Reds", range_color=(0, 1),
        title="Failure rate by business area",
        hover_data=["sessions", "failures", "tests"],
    )
    fig.update_layout(coloraxis_showscale=False)
    return _fig_to_div(fig, "feature-area-chart",
                       yaxis=dict(automargin=True, title=None),
                       xaxis=dict(tickformat=".0%", title="failure rate"))


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
    triage: dict | None = None,
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
    ds = triage or {}

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
        min_builds=settings.flakiness_min_builds,
        cluster_chart=_cluster_chart(clusters),
        device_heatmap=_device_heatmap(device_risk),
        status_donut=_status_donut(metrics),
        trend_chart=_trend_chart(trend),
        category_chart=_category_chart(categories),
        category_rows=category_rows,
        category_details=_category_details(classified, clusters),
        # --- data-science layer ---
        findings=_finding_cards(ds.get("findings")),
        feature_area_chart=_feature_area_chart(ds.get("feature_area_health")),
        feature_area_rows=_records(ds.get("feature_area_health")),
        locator_rows=_records(ds.get("locator_hotspots")),
        duration_rows=_records(ds.get("duration_outliers"), limit=6),
        time_split=ds.get("time_split") or {},
        alpha=settings.inference_alpha,
        perf_rows=_records(ds.get("perf_hotspots")),
        perf_findings=_records(ds.get("perf_vs_failure")),
        platform_rows=_records(ds.get("platform_breakdown")),
        area_by_platform=_records(ds.get("area_by_platform")),
        platforms=_platform_names(ds.get("area_by_platform")),
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
