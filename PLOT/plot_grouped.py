# plot_grouped.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUM_BINS = 500
VIDEO_LEN = 48
LW = 1.5

DEFAULT_SOURCE = r".\DATASET\artifacts\norway\results.all.parquet"
DEFAULT_OUTPUT_DIR = "./graphs"

DEFAULT_GROUPS = ["tram", "car", "bus", "ferry", "metro", "train"]
DEFAULT_ALGOS = ["bb", "bola", "mpc", "rl", "ppo", "netllm"]
DEFAULT_PLOTS = ["tradeoff", "smoothness", "bitrate", "stall", "qoe"]

DEFAULT_LABELS = {
    "bb": "BBA",
    "bola": "BOLA",
    "mpc": "RobustMPC",
    "rl": "Pensieve",
    "ppo": "Pen-PPO",
    "netllm": "NetLLM",
}

DEFAULT_COLORS = {
    "bb": "#4E79A7",
    "bola": "#F28E2B",
    "mpc": "#E15759",
    "rl": "#21EBDA",
    "ppo": "#59A14F",
    "netllm": "#FA12B4",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate grouped ABR comparison plots from txt logs or parquet summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Source folder of .txt logs or a .parquet summary file",
    )
    ap.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write output plots",
    )
    ap.add_argument(
        "--algo",
        nargs="+",
        default=None,
        help="Algorithms to include; if omitted, all default algorithms are used",
    )
    ap.add_argument(
        "--group",
        nargs="+",
        default=None,
        help="Transport groups to include; if omitted, all default groups are used",
    )
    ap.add_argument(
        "--plot",
        nargs="+",
        default=["all"],
        choices=["all", "tradeoff", "smoothness", "bitrate", "stall", "qoe"],
        help="Plot types to generate",
    )
    ap.add_argument(
        "--include-all",
        action="store_true",
        help="Also generate aggregate plot(s) across all groups using suffix 'all'",
    )
    ap.add_argument(
        "--video-len",
        type=float,
        default=VIDEO_LEN,
        help="Video length in seconds used for stall percentage calculation for txt logs",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search source folder recursively for .txt logs",
    )
    return ap.parse_args()


def normalize_plots(plot_names: list[str]) -> list[str]:
    return DEFAULT_PLOTS.copy() if "all" in plot_names else plot_names


def build_labels(algos: list[str]) -> dict[str, str]:
    return {algo: DEFAULT_LABELS.get(algo, algo.upper()) for algo in algos}


def build_colors(algos: list[str]) -> dict[str, str]:
    fallback_palette = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#21EBDA",
        "#59A14F",
        "#FA12B4",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]
    colors: dict[str, str] = {}
    for i, algo in enumerate(algos):
        colors[algo] = DEFAULT_COLORS.get(algo, fallback_palette[i % len(fallback_palette)])
    return colors


def init_metrics(schemes: list[str]) -> dict[str, dict[str, list[float]]]:
    return {
        scheme: {"reward": [], "bitrate": [], "stall": [], "smoothness": []}
        for scheme in schemes
    }


def parse_log_file(path: Path) -> dict[str, list[float]]:
    bitrate: list[float] = []
    rebuffer: list[float] = []
    reward: list[float] = []
    time_all: list[float] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            sp = line.split()
            if len(sp) <= 1:
                continue
            try:
                time_all.append(float(sp[0]))
                bitrate.append(float(sp[1]) / 1000.0)  # kbps -> Mbps
                rebuffer.append(float(sp[3]))
                reward.append(float(sp[-1]))
            except (ValueError, IndexError):
                continue

    return {
        "time": time_all,
        "bitrate": bitrate,
        "rebuffer": rebuffer,
        "reward": reward,
    }


def summarize_metrics(parsed: dict[str, list[float]], video_len: float) -> Optional[dict[str, float]]:
    bitrate = parsed["bitrate"]
    rebuffer = parsed["rebuffer"]
    reward = parsed["reward"]

    if len(bitrate) == 0 or len(reward) <= 1:
        return None

    stall_after_start = float(np.sum(rebuffer[1:]))
    total_time = video_len * 4.0 + stall_after_start

    return {
        "mean_reward": float(np.mean(reward[1:])),
        "mean_bitrate": float(np.mean(bitrate)),
        "stall_pct": float(stall_after_start / total_time * 100.0) if total_time > 0 else 0.0,
        "smoothness": float(np.mean(np.abs(np.diff(bitrate)))) if len(bitrate) > 1 else 0.0,
    }


def safe_trace_index(path: Path) -> tuple[int, str]:
    stem = path.stem
    last = stem.split("_")[-1]
    try:
        return int(last), stem
    except ValueError:
        return 10**9, stem


def collect_scheme_metrics_from_logs(
    root_dir: str | Path,
    schemes: list[str],
    transport: Optional[str],
    video_len: float,
    recursive: bool = True,
) -> dict[str, dict[str, list[float]]]:
    metrics = init_metrics(schemes)

    root = Path(root_dir)
    if not root.exists():
        return metrics

    files = root.rglob("*.txt") if recursive else root.glob("*.txt")
    matched_files = [
        path
        for path in files
        if transport is None or f"_{transport}_" in path.stem.lower()
    ]
    matched_files = sorted(matched_files, key=safe_trace_index)

    for path in matched_files:
        stem_lower = path.stem.lower()
        for scheme in schemes:
            if scheme in stem_lower:
                summary = summarize_metrics(parse_log_file(path), video_len=video_len)
                if summary is None:
                    break
                metrics[scheme]["reward"].append(summary["mean_reward"])
                metrics[scheme]["bitrate"].append(summary["mean_bitrate"])
                metrics[scheme]["stall"].append(summary["stall_pct"])
                metrics[scheme]["smoothness"].append(summary["smoothness"])
                break

    return metrics


def parse_filename_tokens(name: str) -> tuple[Optional[str], Optional[str]]:
    """
    Example:
        log_sim_bb_norway_bus_1
    -> algo='bb', group='bus'
    """
    parts = str(name).lower().replace(".txt", "").split("_")

    algo = None
    group = None

    # Preferred expected pattern:
    # log_sim_<algo>_norway_<group>_<traceid>
    if len(parts) >= 6 and parts[0] == "log" and parts[1] == "sim":
        algo = parts[2]
        group = parts[4]
        return algo, group

    known_groups = set(DEFAULT_GROUPS)

    for token in parts:
        if token in known_groups:
            group = token
            break

    for token in parts:
        if token in set(DEFAULT_ALGOS):
            algo = token
            break

    return algo, group


def find_metric_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def collect_scheme_metrics_from_parquet(
    parquet_path: str | Path,
    schemes: list[str],
    transport: Optional[str],
) -> dict[str, dict[str, list[float]]]:
    metrics = init_metrics(schemes)
    df = pd.read_parquet(parquet_path)

    if df.empty:
        return metrics
    
    #print([ df.head(1)[i] for i in df.columns])

    if "filename" not in df.columns:
        raise ValueError("Parquet file must contain a 'filename' column")

    reward_col = find_metric_column(df, ['6'])
    bitrate_col = find_metric_column(df, ['1'])
    stall_col = find_metric_column(df, ['5'])
    smoothness_col = find_metric_column(df, ["smoothness", "mean_smoothness", "avg_smoothness"])

    work = df.copy()
    work["_algo"] = work["filename"].astype(str).map(lambda x: parse_filename_tokens(x)[0])
    work["_group"] = work["filename"].astype(str).map(lambda x: parse_filename_tokens(x)[1])

    if transport is not None:
        work = work[work["_group"] == transport.lower()]

    for scheme in schemes:
        sub = work[work["_algo"] == scheme.lower()]
        if sub.empty:
            continue

        if reward_col is not None:
            metrics[scheme]["reward"] = pd.to_numeric(sub[reward_col], errors="coerce").dropna().tolist()
        if bitrate_col is not None:
            metrics[scheme]["bitrate"] = pd.to_numeric(sub[bitrate_col], errors="coerce").dropna().tolist()
        if stall_col is not None:
            metrics[scheme]["stall"] = pd.to_numeric(sub[stall_col], errors="coerce").dropna().tolist()
        if smoothness_col is not None:
            metrics[scheme]["smoothness"] = pd.to_numeric(sub[smoothness_col], errors="coerce").dropna().tolist()

    return metrics


def collect_scheme_metrics(
    source: str | Path,
    schemes: list[str],
    transport: Optional[str],
    video_len: float,
    recursive: bool = True,
) -> dict[str, dict[str, list[float]]]:
    source_path = Path(source)

    if source_path.is_file() and source_path.suffix.lower() == ".parquet":
        return collect_scheme_metrics_from_parquet(source_path, schemes, transport)

    if source_path.is_dir():
        return collect_scheme_metrics_from_logs(source_path, schemes, transport, video_len, recursive)

    raise ValueError(f"--source must be an existing folder or a parquet file: {source_path}")


def fix_axes(
    ax,
    xlabel: str,
    ylabel: str,
    title_suffix: str = "",
    invert_x: bool = False,
    invert_y: bool = False,
    xlim=None,
    ylim=None,
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(linestyle="--", linewidth=1.0, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title_suffix:
        ax.set_title(title_suffix)

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=10, ncol=2, edgecolor="white", loc="upper right")


def plot_series(
    vals: dict[str, dict[str, list[float]]],
    schemes: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    metric: str,
    xlabel: str,
    ylabel: str,
    skip: int,
    output_path: str,
    invert_x: bool = False,
    invert_y: bool = False,
    xlim=None,
    ylim=None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False

    for scheme in schemes:
        ys = vals[scheme][metric]
        if not ys:
            continue

        any_data = True
        avg_y = float(np.mean(ys))
        ys_down = ys[::skip] if skip > 1 else ys
        xs = np.arange(len(ys_down))

        ax.plot(
            xs,
            ys_down,
            color=colors[scheme],
            label=f"{labels[scheme]} (Mean: {avg_y:.2f})",
            linewidth=1.5,
            antialiased=True,
            zorder=3,
        )

    if not any_data:
        plt.close(fig)
        return

    fix_axes(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        invert_x=invert_x,
        invert_y=invert_y,
        xlim=xlim,
        ylim=ylim,
    )

    if ax.lines:
        max_x = max(len(line.get_xdata()) for line in ax.lines)
        if max_x > 0:
            step = 5 if max_x > 5 else 1
            ax.set_xticks(np.arange(0, max_x, step))

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_qoe_cdf(
    metrics: dict[str, dict[str, list[float]]],
    schemes: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    output_path: str,
    title_suffix: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    any_data = False

    for scheme in schemes:
        rewards = metrics[scheme]["reward"]
        if not rewards:
            continue

        any_data = True
        values, base = np.histogram(rewards, bins=NUM_BINS)
        cumulative = np.cumsum(values)
        if cumulative[-1] == 0:
            continue
        cumulative = cumulative / cumulative[-1]

        ax.plot(
            base[:-1],
            cumulative,
            "-",
            color=colors[scheme],
            lw=LW,
            label=f"{labels[scheme]}: {np.mean(rewards):.2f}",
        )

    if not any_data:
        plt.close(fig)
        return

    fix_axes(ax, xlabel="QoE", ylabel="CDF", title_suffix=title_suffix)
    ax.set_ylim(0.0, 1.01)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_tradeoff_scatter(
    metrics: dict[str, dict[str, list[float]]],
    schemes: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    output_path: str,
    title: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False

    for scheme in schemes:
        x_vals = metrics[scheme]["bitrate"]
        y_vals = metrics[scheme]["stall"]
        if not x_vals or not y_vals:
            continue

        any_data = True
        mx, my = float(np.mean(x_vals)), float(np.mean(y_vals))
        lx, hx = np.percentile(x_vals, [25, 75])
        ly, hy = np.percentile(y_vals, [25, 75])

        xerr = np.array([[max(0.0, mx - lx)], [max(0.0, hx - mx)]])
        yerr = np.array([[max(0.0, my - ly)], [max(0.0, hy - my)]])

        ax.errorbar(
            mx,
            my,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=colors[scheme],
            capsize=4,
            linewidth=1.5,
            label=f"{labels[scheme]} ({mx:.2f}, {my:.2f})",
        )

    if not any_data:
        plt.close(fig)
        return

    fix_axes(ax, xlabel="Bitrate (Mbps)", ylabel="Stall Ratio (%)", title_suffix=title or "")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def group_plots(
    source: str | Path,
    outdir: Path,
    schemes: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    plots: list[str],
    transport: Optional[str] = None,
    video_len: float = VIDEO_LEN,
    recursive: bool = True,
) -> None:
    metrics = collect_scheme_metrics(
        source=source,
        schemes=schemes,
        transport=transport,
        video_len=video_len,
        recursive=recursive,
    )

    suffix = transport if transport else "all"
    skip = 3 if transport is None else 1

    if "tradeoff" in plots:
        plot_tradeoff_scatter(
            metrics=metrics,
            schemes=schemes,
            labels=labels,
            colors=colors,
            output_path=str(outdir / f"baselines-{suffix}-tradeoff.png"),
            title=f"{suffix.upper()}",
        )

    if "smoothness" in plots:
        plot_series(
            vals=metrics,
            schemes=schemes,
            labels=labels,
            colors=colors,
            metric="smoothness",
            xlabel="Traces",
            ylabel="Bitrate Smoothness",
            skip=skip,
            output_path=str(outdir / f"baselines-{suffix}-sr.png"),
        )

    if "bitrate" in plots:
        plot_series(
            vals=metrics,
            schemes=schemes,
            labels=labels,
            colors=colors,
            metric="bitrate",
            xlabel="Traces",
            ylabel="Video Bitrate (Mbps)",
            skip=skip,
            output_path=str(outdir / f"baselines-{suffix}-br.png"),
        )

    if "stall" in plots:
        plot_series(
            vals=metrics,
            schemes=schemes,
            labels=labels,
            colors=colors,
            metric="stall",
            xlabel="Traces",
            ylabel="Time Spent on Stall (%)",
            skip=skip,
            output_path=str(outdir / f"baselines-{suffix}-st.png"),
        )

    if "qoe" in plots:
        plot_qoe_cdf(
            metrics=metrics,
            schemes=schemes,
            labels=labels,
            colors=colors,
            output_path=str(outdir / f"baselines-{suffix}-qoe.png"),
            title_suffix=suffix.upper(),
        )


def main() -> None:
    args = parse_args()

    source = Path(args.source)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    plots = normalize_plots(args.plot)
    schemes = args.algo or DEFAULT_ALGOS
    groups = args.group or DEFAULT_GROUPS

    labels = build_labels(schemes)
    colors = build_colors(schemes)

    if args.include_all:
        group_plots(
            source=source,
            outdir=outdir,
            schemes=schemes,
            labels=labels,
            colors=colors,
            plots=plots,
            transport=None,
            video_len=args.video_len,
            recursive=args.recursive,
        )

    for transport in groups:
        group_plots(
            source=source,
            outdir=outdir,
            schemes=schemes,
            labels=labels,
            colors=colors,
            plots=plots,
            transport=transport,
            video_len=args.video_len,
            recursive=args.recursive,
        )


if __name__ == "__main__":
    main()