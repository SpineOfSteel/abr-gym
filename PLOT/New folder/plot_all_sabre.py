from __future__ import annotations

import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import mpld3


PY = "python"
SABRE = "../sab.py"

TMP = Path("tmp")
FIG = Path("figures")
STATS = Path("stats")


# ----------------------------- small utils -----------------------------

def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_text(path: str | Path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")

def run(cmd: Sequence[str], timeout_s: int = 180) -> str:
    print(cmd)
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,   # or PIPE if you need it
            text=True,
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        return p.stdout
    except subprocess.TimeoutExpired:
        return ""  # or return partial stdout if you prefer
        
def save_html(fig, out_html: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    mpld3.save_html(fig, str(out_html))
    plt.close(fig)


# ----------------------------- CDF + stats -----------------------------

def cdf_points(samples: Sequence[float], margin: float = 0.025) -> List[Tuple[float, float]]:
    if not samples:
        return []
    s = sorted(samples)
    span = s[-1] - s[0]
    m = margin * span if span > 0 else margin
    inc = 1.0 / len(s)
    y = 0.0
    out: List[Tuple[float, float]] = []
    if span == 0:
        out.append((s[0] - m, y))
    for x in s:
        out.append((x, y))
        y += inc
        out.append((x, y))
    if span == 0:
        out.append((s[-1] + m, y))
    return out

def median_mean_std(samples: Sequence[float]) -> Tuple[float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0
    s = sorted(samples)
    n = len(s)
    med = (s[(n - 1) // 2] + s[n // 2]) / 2.0
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n
    return med, mean, math.sqrt(var)

def sparse_marker_points(points: Sequence[Tuple[float, float]], k: int, alg_i: int, n_algs: int):
    """Pick ~k points to mark, with slight offset per algorithm (like old .dot)."""
    if not points:
        return [], []
    step = max(1, len(points) // k)
    first = math.ceil(alg_i / (n_algs + 1) * step)
    chunk = points[first::step]
    if not chunk:
        return [], []
    xs, ys = zip(*chunk)
    return list(xs), list(ys)


# ----------------------------- sabre runner -----------------------------

def parse_kv(stdout: str, wanted: Iterable[str]) -> Dict[str, float]:
    wanted = set(wanted)
    out: Dict[str, float] = {}
    for line in stdout.splitlines():
        k, sep, v = line.partition(":")
        if not sep:
            continue
        k = k.strip()
        if k in wanted:
            try:
                out[k] = float(v.strip())
            except ValueError:
                pass
    return out

def sabre_one_trace(trace_path: Path, args: Sequence[str], metric_keys: Sequence[str]) -> Dict[str, float]:
    cmd = [PY, SABRE, "-n", str(trace_path), *args]
    return parse_kv(run(cmd), metric_keys)


# ----------------------------- configs -----------------------------

@dataclass(frozen=True)
class Metric:
    name: str
    key: str
    cfg: Dict[str, object] = field(default_factory=dict)  # xtics, xoffset, etc.

@dataclass(frozen=True)
class Algorithm:
    name: str
    sabre_args: List[str]
    show_legend: bool = True

@dataclass(frozen=True)
class Subfig:
    title: str
    trace_dir: str
    base_args: List[str]
    xranges: Optional[List[float]] = None  # per-metric x max, same order as metrics


def xlabel_from(metric_name: str) -> str:
    if "time" in metric_name:
        return f"{metric_name} (s)"
    if "bitrate" in metric_name:
        return f"{metric_name} (kbps)"
    return metric_name


# ----------------------------- combined multi-axes CDF plot -----------------------------

def do_figure_mpld3(
    outname: str,
    subfigs: Sequence[Subfig],
    algorithms: Sequence[Algorithm],
    metrics: Sequence[Metric],
    *,
    max_workers: int = 5,
    font_size: int = 16,
) -> None:
    """
    Produces:
      figures/{outname}.html  (ONE figure, many axes)
      stats/{outname}-{subfig}.txt
    """
    plt.rcParams.update({"font.size": font_size})

    nrows, ncols = len(subfigs), len(metrics)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    # For a clean look: one legend for the whole figure.
    legend_handles = None
    legend_labels = None

    for r, sub in enumerate(subfigs):
        traces = [p for p in sorted(Path(sub.trace_dir).iterdir()) if p.is_file()]
        metric_keys = [m.key for m in metrics]

        # info[metric_name][alg_name] = (median, mean, std, points)
        info: Dict[str, Dict[str, Tuple[float, float, float, List[Tuple[float, float]]]]] = {
            m.name: {} for m in metrics
        }

        for alg_i, alg in enumerate(algorithms, start=1):
            buckets: Dict[str, List[float]] = {m.key: [] for m in metrics}
            args = [*sub.base_args, *alg.sabre_args]

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(sabre_one_trace, tr, args, metric_keys) for tr in traces]
                for fut in as_completed(futs):
                    res = fut.result()
                    for k, v in res.items():
                        buckets[k].append(v)

            for m in metrics:
                cfg = m.cfg or {}
                xoff = float(cfg.get("xoffset", 0.0))
                samples = buckets[m.key]
                pts = [(x + xoff, y) for x, y in cdf_points(samples)]
                med, mean, sd = median_mean_std(samples)
                info[m.name][alg.name] = (med, mean, sd, pts)

        # stats file per subfig (same as before, but keyed to outname)
        stat_file = STATS / f"{outname}-{sub.title.replace(' ','-')}.txt"
        lines: List[str] = []
        for m in metrics:
            lines.append(f"{m.name}:")
            for alg in algorithms:
                med, mean, sd, _ = info[m.name][alg.name]
                lines.append(f"{alg.name}: {med} {mean} {sd}")
            lines.append("")
        write_text(stat_file, "\n".join(lines).rstrip() + "\n")

        # plot each metric in its column
        for c, m in enumerate(metrics):
            ax = axes[r][c]
            cfg = m.cfg or {}

            ax.set_title(f"{sub.title} — {m.name}")
            ax.set_ylim(0, 1)
            if r == nrows - 1:
                ax.set_xlabel(xlabel_from(m.name))
            if c == 0:
                ax.set_ylabel("CDF")

            # xlim per-subfig per-metric if provided
            if sub.xranges and c < len(sub.xranges) and sub.xranges[c] is not None:
                ax.set_xlim(0, float(sub.xranges[c]))

            # xtics (if numeric) -> MultipleLocator
            xtics = cfg.get("xtics", None)
            if isinstance(xtics, (int, float)) and xtics > 0:
                ax.xaxis.set_major_locator(MultipleLocator(float(xtics)))

            # algorithm curves
            for alg_i, alg in enumerate(algorithms, start=1):
                _, _, _, pts = info[m.name][alg.name]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                label = alg.name if alg.show_legend else "_nolegend_"
                (line,) = ax.plot(xs, ys, label=label, linewidth=2)
                mx, my = sparse_marker_points(pts, k=4, alg_i=alg_i, n_algs=len(algorithms))
                if mx:
                    ax.plot(mx, my, linestyle="None", marker="o", markersize=4, color=line.get_color())

            # capture legend from first axis only
            if legend_handles is None and legend_labels is None:
                handles, labels = ax.get_legend_handles_labels()
                # filter out no-legend labels
                kept = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
                if kept:
                    legend_handles, legend_labels = zip(*kept)

            ax.legend_.remove() if ax.get_legend() else None

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(legend_labels),
            frameon=False,
        )

    fig.tight_layout()
    save_html(fig, FIG / f"{outname}.html")


# ----------------------------- combined fig6 + fig1/4 (multi-axes) -----------------------------

def write_network_json(path: Path, bw_kbps: int = 8000, dur_ms: int = 60000, lat_ms: int = 0) -> None:
    write_text(path, f'[ {{"duration_ms": {dur_ms}, "bandwidth_kbps": {bw_kbps}, "latency_ms": {lat_ms}}} ]')

def parse_verbose_quality_series(stdout: str, bitrates_kbps: Sequence[int],
                                start_idx: int = 0, end_idx: Optional[int] = None,
                                gap_at: Optional[int] = None) -> List[Optional[Tuple[float, float]]]:
    series: List[Optional[Tuple[float, float]]] = []
    for line in stdout.splitlines():
        if "[" not in line or "Network" in line:
            continue
        parts = line.split()
        idx = int(parts[1].split(":")[0])
        q = int(parts[2].split("=")[1])
        if idx < start_idx:
            continue
        if gap_at is not None and idx == gap_at:
            series.append(None)
        series.append((idx * 3, bitrates_kbps[q]))
        series.append(((idx + 1) * 3, bitrates_kbps[q]))
        if end_idx is not None and idx >= end_idx:
            break
    return series

def plot_step_series(ax, series, *, label: str, dashed: bool = False, show_label: bool = True):
    # split on None to create gaps
    chunk_x, chunk_y = [], []
    first = True
    for p in list(series) + [None]:
        if p is None:
            if chunk_x:
                ax.plot(
                    chunk_x,
                    chunk_y,
                    linewidth=2,
                    linestyle="--" if dashed else "-",
                    label=(label if (first and show_label) else "_nolegend_"),
                )
                first = False
                chunk_x, chunk_y = [], []
            continue
        chunk_x.append(p[0]); chunk_y.append(p[1])

def figure6_combined(bbb: dict) -> None:
    plt.rcParams.update({"font.size": 16})
    ensure_dirs(TMP)

    write_network_json(TMP / "network.json")

    base_a = [PY, SABRE, "-v", "-m", "bbb.json", "-n", str(TMP / "network.json")]
    base_b = [PY, SABRE, "-v", "-m", "bbb.json", "-n", str(TMP / "network.json"), "-s", "120", "180"]

    out_bola_a = run([*base_a, "-a", "bola", "-ab"])
    out_pl_a   = run([*base_a, "-a", "bolae"])

    out_bola_b = run([*base_b, "-a", "bola", "-ab"])
    out_pl_b   = run([*base_b, "-a", "bolae"])

    s1a = parse_verbose_quality_series(out_bola_a, bbb["bitrates_kbps"], start_idx=0, end_idx=9)
    s2a = parse_verbose_quality_series(out_pl_a,   bbb["bitrates_kbps"], start_idx=0, end_idx=9)

    s1b = parse_verbose_quality_series(out_bola_b, bbb["bitrates_kbps"], start_idx=35, end_idx=69, gap_at=60)
    s2b = parse_verbose_quality_series(out_pl_b,   bbb["bitrates_kbps"], start_idx=35, end_idx=69, gap_at=60)

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), squeeze=False)
    ax1, ax2 = axes[0][0], axes[1][0]

    # 6a
    ax1.set_title("fig6a")
    ax1.set_xlabel("play time (s)")
    ax1.set_ylabel("bitrate (kbps)")
    ax1.set_ylim(0, 6500)
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    plot_step_series(ax1, s1a, label="BOLA", dashed=True, show_label=True)
    plot_step_series(ax1, s2a, label="BOLA-PL", dashed=False, show_label=False)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

    # 6b
    ax2.set_title("fig6b")
    ax2.set_xlabel("play time (s)")
    ax2.set_ylabel("bitrate (kbps)")
    ax2.set_ylim(0, 6500)
    ax2.set_xlim(180, max(p[0] for p in s2b if p))
    ax2.xaxis.set_major_locator(MultipleLocator(10))
    plot_step_series(ax2, s1b, label="BOLA", dashed=True, show_label=False)
    plot_step_series(ax2, s2b, label="BOLA-PL", dashed=False, show_label=True)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

    fig.tight_layout()
    save_html(fig, FIG / "fig6.html")

def figure_1_4_combined() -> None:
    plt.rcParams.update({"font.size": 16})

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), squeeze=False)
    ax1, ax2, ax3 = axes[0]

    # fig-1
    ax1.set_title("fig-1")
    ax1.set_xlim(0, 18); ax1.set_ylim(0, 6000)
    ax1.set_xlabel("buffer level (s)")
    ax1.set_ylabel("bitrate (kbps)")
    x = [0, 5, 5, 10, 10, 15, 15, 18]
    y = [1000, 1000, 2500, 2500, 5000, 5000, 0, 0]
    ax1.plot(x, y, linewidth=2)
    ax1.axvline(5, linestyle="--", linewidth=1)
    ax1.axvline(10, linestyle="--", linewidth=1)
    ax1.hlines([2500, 5000], xmin=0, xmax=[5, 10], linestyles="--", linewidth=1)

    # fig-4a
    ax2.set_title("fig-4a")
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 6500)
    ax2.set_xlabel("buffer level (s)")
    ax2.set_ylabel("bitrate (kbps)")
    pts_a = [
        (0,0),(0,230),(3.534,230),(3.534,331),(3.843,331),(3.843,477),(4.153,477),(4.153,688),
        (4.462,688),(4.462,991),(4.771,991),(4.771,1427),(5.081,1427),(5.081,2056),(5.390,2056),
        (5.390,2962),(5.759,2962),(5.759,5027),(6.075,5027),(6.075,6000),(7.0,6000),(7.0,0),(10,0)
    ]
    ax2.plot([p[0] for p in pts_a], [p[1] for p in pts_a], linewidth=2)

    # fig-4b
    ax3.set_title("fig-4b")
    ax3.set_xlim(0, 40); ax3.set_ylim(0, 6500)
    ax3.set_xlabel("buffer level (s)")
    ax3.set_ylabel("bitrate (kbps)")
    pts_b = [
        (0,0),(0,230),(11.048,230),(11.048,331),(13.284,331),(13.284,477),(15.527,477),(15.527,688),
        (17.770,688),(17.770,991),(20.007,991),(20.007,1427),(22.244,1427),(22.244,2056),(24.483,2056),
        (24.483,2962),(27.150,2962),(27.150,5027),(29.441,5027),(29.441,6000),(36.132,6000),(36.132,0),(40,0)
    ]
    ax3.plot([p[0] for p in pts_b], [p[1] for p in pts_b], linewidth=2)

    ax3.add_patch(Rectangle((16, 0), 10, 6500, alpha=0.25))
    ax3.annotate("", xy=(15.9, 3500), xytext=(0.1, 3500), arrowprops=dict(arrowstyle="<->"))
    ax3.annotate("", xy=(25.9, 3500), xytext=(16.1, 3500), arrowprops=dict(arrowstyle="<->"))
    ax3.text(8, 5600, "virtual\nplaceholder\nsegments", ha="center", va="center")
    ax3.text(21, 5600, "actual\nvideo\nsegments", ha="center", va="center")

    fig.tight_layout()
    save_html(fig, FIG / "fig-1_4.html")


# ----------------------------- your existing figure wrappers (updated) -----------------------------

def figure_7_10() -> None:
    subfigs = [Subfig("4G VOD", "4Glogs", ["-m", "bbb4k.json", "-b", "25"])]

    metrics = [Metric("reaction time", "rampup time", {"xtics": 10})]

    do_figure_mpld3(
        "fig7a",
        subfigs,
        [Algorithm("BOLA", ["-ao", "-a", "bola", "-ab"], True),
         Algorithm("BOLA-PL", ["-ao", "-a", "bolae", "-noibr"], False)],
        metrics,
    )

    do_figure_mpld3(
        "fig7b",
        subfigs,
        [Algorithm("BOLA", ["-ao", "-a", "bola", "-ab", "-s", "120", "180"], False),
         Algorithm("BOLA-PL", ["-ao", "-a", "bolae", "-noibr", "-s", "120", "180"], True)],
        metrics,
    )

    do_figure_mpld3(
        "fig10a",
        subfigs,
        [Algorithm("BOLA", ["-ao", "-a", "bola", "-ab"]),
         Algorithm("TPUT", ["-ao", "-a", "throughput"]),
         Algorithm("DYNAMIC", ["-ao", "-a", "dynamic", "-ab"], False)],
        metrics,
    )

    do_figure_mpld3(
        "fig10b",
        subfigs,
        [Algorithm("BOLA", ["-ao", "-a", "bola", "-ab", "-s", "120", "180"], False),
         Algorithm("TPUT", ["-ao", "-a", "throughput", "-s", "120", "180"], False),
         Algorithm("DYNAMIC", ["-ao", "-a", "dynamic", "-ab", "-s", "120", "180"], True)],
        metrics,
    )

def figure8() -> None:
    do_figure_mpld3(
        "fig8",
        [Subfig("3G Live 10s", "3Glogs", ["-m", "bbb.json", "-b", "10"], [0.6, 600, 2000])],
        [Algorithm("BOLA", ["-a", "bola", "-ao", "-ab"]),
         Algorithm("BOLA-PL", ["-a", "bolae", "-ao", "-noibr"]),
         Algorithm("BOLA-E", ["-a", "bolae", "-ao"])],
        [Metric("rebuffer ratio", "rebuffer ratio", {"xtics": 0.1}),
         Metric("average bitrate oscillation", "time average bitrate change", {"xtics": 150}),
         Metric("average bitrate", "time average played bitrate", {"xtics": 500})],
    )

def figure11() -> None:
    do_figure_mpld3(
        "fig11",
        [Subfig("4G VOD", "4Glogs", ["-m", "bbb4k.json", "-b", "25"], [0.1, 2200, 34000]),
         Subfig("4G Live 10s", "4Glogs", ["-m", "bbb4k.json", "-b", "10"], [0.1, 4600, 31500])],
        [Algorithm("BOLA", ["-ao", "-a", "bola"]),
         Algorithm("THROUGHPUT", ["-ao", "-a", "throughput"]),
         Algorithm("DYNAMIC", ["-ao", "-a", "dynamic"])],
        [Metric("rebuffer ratio", "rebuffer ratio"),
         Metric("average bitrate oscillation", "time average bitrate change", {"xtics": 1000}),
         Metric("average bitrate", "time average played bitrate", {"xtics": 10000})],
    )

def figure_12_13() -> None:
    do_figure_mpld3(
        "12_13_SD",
        [Subfig("FCC SD", "sd_fs", ["-m", "bbb.json", "-b", "25"], [0.01, 450, 4500, 120])],
        [Algorithm("BOLA-E", ["-ao", "-r", "none", "-a", "bolae", "-rmp", "9", "-ml", "180"]),
         Algorithm("BOLA-E-FS", ["-ao", "-r", "left", "-a", "bolae", "-rmp", "9", "-ml", "180"]),
         Algorithm("DYNAMIC", ["-ao", "-r", "none", "-a", "dynamic", "-rmp", "9", "-ml", "180"], False),
         Algorithm("DYNAMIC-FS", ["-ao", "-r", "left", "-a", "dynamic", "-rmp", "9", "-ml", "180"], False)],
        [Metric("rebuffer ratio", "rebuffer ratio"),
         Metric("average bitrate oscillation", "time average bitrate change", {"xtics": 150}),
         Metric("average bitrate", "time average played bitrate", {"xtics": 1500}),
         Metric("reaction time", "rampup time", {"xoffset": -60, "xtics": 40})],
    )

    do_figure_mpld3(
        "12_13_HD",
        [Subfig("FCC HD", "hd_fs", ["-m", "bbb4k.json", "-b", "25"], [0.01, 1200, 12000, 120])],
        [Algorithm("BOLA-E", ["-ao", "-r", "none", "-a", "bolae", "-rmp", "4", "-ml", "180"], False),
         Algorithm("BOLA-E-FS", ["-ao", "-r", "left", "-a", "bolae", "-rmp", "4", "-ml", "180"], False),
         Algorithm("DYNAMIC", ["-ao", "-r", "none", "-a", "dynamic", "-rmp", "4", "-ml", "180"]),
         Algorithm("DYNAMIC-FS", ["-ao", "-r", "left", "-a", "dynamic", "-rmp", "4", "-ml", "180"])],
        [Metric("rebuffer ratio", "rebuffer ratio"),
         Metric("average bitrate oscillation", "time average bitrate change", {"xtics": 400}),
         Metric("average bitrate", "time average played bitrate", {"xtics": 4000}),
         Metric("reaction time", "rampup time", {"xoffset": -60, "xtics": 40})],
    )


# ----------------------------- main -----------------------------

def main() -> None:
    ensure_dirs(TMP, FIG, STATS)

    bbb = load_json("bbb.json")

    figure6_combined(bbb); print("DONE fig6.html")
    #figure_1_4_combined(); print("DONE fig-1_4.html")
    #figure_7_10(); print("DONE fig7a/fig7b/fig10a/fig10b")
    #figure8(); print("DONE fig8.html")
    #figure11(); print("DONE fig11.html")
    #figure_12_13(); print("DONE 12_13_SD/12_13_HD")

if __name__ == "__main__":
    main()
