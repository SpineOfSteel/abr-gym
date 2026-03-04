from pathlib import Path
import math
import re
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mpld3
from mpld3 import plugins

# -----------------------------
# Existing log-report settings
# -----------------------------
OUT = Path("./DATASET/TRACES/norway_tram/")
PAT = "log_*.txt"
LAST_N = 300   # set None to plot all samples
HTML = OUT / "abr_report.html"

# Optional QUETRA-style aggregate input
RESULTS_CSV = Path("results.csv")

LABELS = {
    "bitrate": "Average Bitrate (kbps)",
    "change": "Changes in Representation",
    "ineff": "Inefficiency",
    "stall": "Stall Duration (Sec)",
    "numStall": "Number of Stalls",
    "avgStall": "Average Stall Duration (Sec)",
    "overflow": "Buffer Full Duration (Sec)",
    "numOverflow": "Number of Buffer Overflow",
    "qoe": "QoE (x100,000)",
}

METHODNAME = {
    "abr": "Dash.js ABR",
    "elastic": "ELASTIC",
    "qAvgTh": "Avg Th",
    "bba": "BBA",
    "quetra": "QUETRA",
    "bola": "BOLA",
    "qLowPassEMA": "Low Pass EMA",
    "qGradientEMA": "Gradient EMA",
    "qKAMA": "KAMA",
    "qEMA": "EMA",
}

SMOOTHING_METHODS = ["qAvgTh", "qEMA", "quetra", "qKAMA", "qGradientEMA", "qLowPassEMA"]
ADAPTATION_METHODS = ["abr", "bola", "elastic", "bba", "quetra"]
BUF_ORDER = ["30/60", "120", "240"]
BUF_LABELS = {"30/60": "30/60 s", "120": "120 s", "240": "240 s"}


def read_log(p: Path):
    rows = []
    for ln in p.read_text(errors="ignore").splitlines():
        sp = ln.split()
        if len(sp) < 7:
            continue
        try:
            rows.append([float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]), float(sp[6])])
        except Exception:
            pass
    if not rows:
        return None
    a = np.asarray(rows, float)
    t = a[:, 0] - a[0, 0]
    br, buf, rb, rew = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    if LAST_N is not None and len(t) > LAST_N:
        sl = slice(-LAST_N, None)
        t, br, buf, rb, rew = t[sl], br[sl], buf[sl], rb[sl], rew[sl]
    return t, br, buf, rb, rew


def algo_name(p: Path):
    parts = p.name.split("_")
    return parts[2] if len(parts) > 2 else p.stem


def overlay(data, kind, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    lines, labels = [], []
    for algo, (t, br, buf, rb, rew) in sorted(data.items()):
        y = {"reward": rew, "bitrate": br, "buffer": buf, "rebuffer": rb}[kind]
        ln = ax.plot(t, y, lw=1, label=algo)[0]
        lines.append(ln)
        labels.append(algo)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    plugins.connect(fig, plugins.InteractiveLegendPlugin(lines, labels, alpha_unsel=0.15))
    return fig


def stats_row(algo, d):
    _, br, buf, rb, rew = d
    return (algo, len(rew), float(np.mean(rew)), float(np.mean(br)), float(np.mean(buf)), float(np.sum(rb)))


# -----------------------------
# R -> Python conversion helpers
# -----------------------------
def check_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prep_methods(df: pd.DataFrame, methods, quetra_label=None):
    out = df.copy()
    mapping = METHODNAME.copy()
    if quetra_label is not None:
        mapping["quetra"] = quetra_label
    out = out[out["method"].isin(methods)].copy()
    out["method_label"] = out["method"].map(mapping).fillna(out["method"])
    out["method_label"] = pd.Categorical(
        out["method_label"],
        categories=[mapping.get(m, m) for m in methods],
        ordered=True,
    )
    out["bufSize"] = out["bufSize"].astype(str)
    out["bufSize_label"] = pd.Categorical(
        out["bufSize"].map(BUF_LABELS).fillna(out["bufSize"]),
        categories=[BUF_LABELS[b] for b in BUF_ORDER if b in out["bufSize"].astype(str).unique()],
        ordered=True,
    )
    return out


def aggregate_mean(df: pd.DataFrame, group_cols):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg = df.groupby(group_cols, as_index=False)[numeric_cols].mean()
    return agg


def mpld3_html(fig):
    html_text = mpld3.fig_to_html(fig)
    plt.close(fig)
    return html_text


def scatter_facets(df, metricx, metricy, methods, y_limits=None, y_ticks=None, x_limits=None, x_ticks=None, title=None, quetra_label=None):
    check_columns(df, [metricx, metricy, "method", "bufSize"])
    dt = prep_methods(df, methods, quetra_label=quetra_label)
    buf_values = [b for b in BUF_ORDER if b in dt["bufSize"].astype(str).unique()]
    fig, axes = plt.subplots(1, max(1, len(buf_values)), figsize=(7 * max(1, len(buf_values)), 5), squeeze=False)
    axes = axes.ravel()

    method_levels = list(dt["method_label"].cat.categories)
    markers = ['s', '^', 'o', 'D', 'P', 'X', '*', 'v']

    for ax, buf in zip(axes, buf_values):
        sub = dt[dt["bufSize"].astype(str) == buf]
        for i, method in enumerate(method_levels):
            sm = sub[sub["method_label"] == method]
            if sm.empty:
                continue
            ax.scatter(
                sm[metricx], sm[metricy],
                s=140,
                marker=markers[i % len(markers)],
                linewidths=1.8,
                facecolors='none',
                label=str(method),
            )
            ax.scatter(
                sm[metricx], sm[metricy],
                s=45,
                marker='+',
                linewidths=1.8,
            )
        ax.set_title(BUF_LABELS.get(buf, buf))
        ax.set_xlabel(LABELS.get(metricx, metricx))
        ax.set_ylabel(LABELS.get(metricy, metricy))
        ax.grid(True, alpha=0.25)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 6), frameon=False, bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def grouped_bar(df, metric, methods, ylim=None, title=None):
    check_columns(df, [metric, "method", "bufSize"])
    dt = prep_methods(df, methods)
    pivot = dt.pivot_table(index="method_label", columns="bufSize", values=metric, aggfunc="mean")
    pivot = pivot.reindex(index=list(dt["method_label"].cat.categories))
    pivot = pivot[[c for c in BUF_ORDER if c in pivot.columns]]

    if metric == "qoe":
        pivot = pivot / 100000.0
        if ylim is None:
            ylim = (-0.2, 5)
    elif metric == "overflow" and ylim is None:
        ylim = (-3, 80)

    x = np.arange(len(pivot.index))
    width = 0.22 if len(pivot.columns) >= 3 else 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    hatches = ['///', '\\\\', 'xx', '..']

    for i, col in enumerate(pivot.columns):
        ax.bar(x + (i - (len(pivot.columns)-1)/2) * width, pivot[col].values, width=width, label=BUF_LABELS.get(col, col), hatch=hatches[i % len(hatches)], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in pivot.index], rotation=20, ha='right')
    ax.set_ylabel(LABELS.get(metric, metric))
    ax.set_title(title or f"{LABELS.get(metric, metric)} by Method and Buffer Capacity")
    ax.grid(True, axis='y', alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(title="Buffer Capacity (Sec)", frameon=False)
    fig.tight_layout()
    return fig


def load_results_csv(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"method", "bufSize"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain at least columns: {sorted(required)}")
    return df


def build_r_converted_sections(df: pd.DataFrame):
    sections = []

    # fig4a / fig4b from method+bufSize mean
    mean_dt = aggregate_mean(df, ["method", "bufSize"])
    dt = mean_dt[mean_dt["method"].isin(ADAPTATION_METHODS)].copy()

    fig = scatter_facets(
        dt, "change", "bitrate", ADAPTATION_METHODS,
        y_limits=(1500, 2400), y_ticks=np.arange(1500, 2401, 300),
        x_limits=(0, 220), x_ticks=np.arange(0, 201, 50),
        title="Fig 4a: Changes vs Average Bitrate",
    )
    sections.append(("Fig 4a: Changes vs Average Bitrate", mpld3_html(fig)))

    fig = scatter_facets(
        dt, "stall", "numStall", ADAPTATION_METHODS,
        y_limits=(2.4, 5.0), y_ticks=np.arange(2.5, 5.1, 0.5),
        x_limits=(4.5, 21), x_ticks=np.arange(5, 21, 5),
        title="Fig 4b: Stall Duration vs Number of Stalls",
    )
    sections.append(("Fig 4b: Stall Duration vs Number of Stalls", mpld3_html(fig)))

    # fig7 from method+bufSize+sample mean for sample v5
    if "sample" in df.columns:
        mean_dt = aggregate_mean(df, ["method", "bufSize", "sample"])
        dt = mean_dt[(mean_dt["sample"].astype(str) == "v5") & (mean_dt["method"].isin(ADAPTATION_METHODS))].copy()
        if not dt.empty:
            fig = scatter_facets(
                dt, "change", "bitrate", ADAPTATION_METHODS,
                y_limits=(1150, 2200), y_ticks=np.arange(1200, 2201, 250),
                x_limits=(0, 35), x_ticks=np.arange(5, 36, 10),
                title="Fig 7: Sample v5 — Changes vs Average Bitrate",
            )
            sections.append(("Fig 7: Sample v5 — Changes vs Average Bitrate", mpld3_html(fig)))

    # fig9 smoothing methods, rename quetra => Last Th
    dt = mean_dt = aggregate_mean(df, ["method", "bufSize"])
    dt = mean_dt[mean_dt["method"].isin(SMOOTHING_METHODS)].copy()
    fig = scatter_facets(
        dt, "change", "bitrate", SMOOTHING_METHODS,
        y_limits=(1500, 2500), y_ticks=np.arange(1500, 2501, 250),
        x_limits=(0, 65), x_ticks=np.arange(0, 61, 20),
        title="Fig 9: Smoothing Methods — Changes vs Average Bitrate",
        quetra_label="Last Th",
    )
    sections.append(("Fig 9: Smoothing Methods — Changes vs Average Bitrate", mpld3_html(fig)))

    # fig5 and fig8 bar charts
    dt = aggregate_mean(df, ["method", "bufSize"])
    dt = dt[dt["method"].isin(ADAPTATION_METHODS)].copy()

    if "qoe" in dt.columns:
        fig = grouped_bar(dt, "qoe", ADAPTATION_METHODS, ylim=(-0.2, 5), title="Fig 5: QoE by Method and Buffer Capacity")
        sections.append(("Fig 5: QoE by Method and Buffer Capacity", mpld3_html(fig)))

    if "overflow" in dt.columns:
        fig = grouped_bar(dt, "overflow", ADAPTATION_METHODS, ylim=(-3, 80), title="Fig 8: Buffer Overflow Duration by Method and Buffer Capacity")
        sections.append(("Fig 8: Buffer Overflow Duration by Method and Buffer Capacity", mpld3_html(fig)))

    return sections


def main():
    parts = ["""<!doctype html><html><head><meta charset='utf-8'/>
<title>ABR Report</title>
<style>
body{font-family:Arial,sans-serif;margin:20px;line-height:1.4}
table{border-collapse:collapse;width:100%;margin:12px 0 18px}
th,td{border:1px solid #ddd;padding:6px;text-align:left}
th{background:#f6f6f6}
.section{margin:24px 0}
.small{color:#666;font-size:13px}
code{background:#f6f6f6;padding:2px 4px}
</style></head><body>
<h1>ABR Report</h1>
<p class='small'>Contains the original time-series report plus Python conversions of the R plots when <code>results.csv</code> is available.</p>
"""]

    # Original time-series report
    if OUT.exists() and OUT.is_dir():
        files = sorted(OUT.glob(PAT))
        data = {algo_name(p): read_log(p) for p in files}
        data = {k: v for k, v in data.items() if v is not None}
    else:
        data = {}

    if data:
        rows = [stats_row(a, d) for a, d in sorted(data.items())]
        parts.append("""<h2>Summary</h2>
<table><thead><tr>
<th>algo</th><th>n</th><th>avg_reward</th><th>avg_bitrate_kbps</th><th>avg_buffer_s</th><th>total_rebuf_s</th>
</tr></thead><tbody>""")
        for algo, n, ar, ab, aBuf, tr in rows:
            parts.append(f"<tr><td>{html.escape(str(algo))}</td><td>{n}</td><td>{ar:.3f}</td><td>{ab:.1f}</td><td>{aBuf:.3f}</td><td>{tr:.3f}</td></tr>")
        parts.append("</tbody></table>")

        for kind, yl, tt in [
            ("reward", "QoE / Reward", "Reward (toggle algos in legend)"),
            ("bitrate", "Bitrate (Kbps)", "Bitrate"),
            ("buffer", "Buffer (s)", "Buffer"),
            ("rebuffer", "Rebuffer (s)", "Rebuffer"),
        ]:
            fig = overlay(data, kind, yl, tt)
            parts.append(f"<div class='section'><h2>{html.escape(tt)}</h2>{mpld3.fig_to_html(fig)}</div>")
            plt.close(fig)
    else:
        parts.append(f"<p class='small'>No parseable log files found under <code>{html.escape(str(OUT))}</code> matching <code>{html.escape(PAT)}</code>.</p>")

    # Converted R plots
    try:
        results_df = load_results_csv(RESULTS_CSV)
    except Exception as e:
        results_df = None
        parts.append(f"<p class='small'>Could not load <code>{html.escape(str(RESULTS_CSV))}</code>: {html.escape(str(e))}</p>")

    if results_df is not None:
        parts.append("<h2>Converted R-style Aggregate Figures</h2>")
        try:
            for title, fig_html in build_r_converted_sections(results_df):
                parts.append(f"<div class='section'><h2>{html.escape(title)}</h2>{fig_html}</div>")
        except Exception as e:
            parts.append(f"<p class='small'>Failed while creating converted R plots: {html.escape(str(e))}</p>")
    else:
        parts.append("<p class='small'>Place a compatible <code>results.csv</code> next to this script to enable the converted R figures (fig4a, fig4b, fig5, fig7, fig8, fig9).</p>")

    parts.append("</body></html>")
    OUT.mkdir(parents=True, exist_ok=True)
    HTML.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote: {HTML.resolve()}")


if __name__ == "__main__":
    main()
