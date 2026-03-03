
"""
abr_plot_lib.py


Focus:
- one parser for multiple log styles
- centralized constants/config
- PNG export helpers
- reusable plotting API
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------- config -----------------------------

@dataclass
class QoEConfig:
    rebuffer_penalty: float = 4.3
    smooth_penalty: float = 1.0
    kbps_per_mbps: float = 1000.0


@dataclass
class ABRPlotConfig:
    results_dir: Path | str = Path("./results")
    output_dir: Path | str = Path("./figures")
    video_len: Optional[int] = None
    num_bins: int = 100
    bits_in_byte: float = 8.0
    millisec_in_sec: float = 1000.0
    million: float = 1_000_000.0
    qoe: QoEConfig = field(default_factory=QoEConfig)
    video_bitrates_kbps: Optional[List[float]] = None
    scheme_aliases: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.results_dir = Path(self.results_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def canonical_scheme(self, raw: str) -> str:
        return self.scheme_aliases.get(raw, raw)


DEFAULT_SCHEME_ORDER = [
    "BB", "RB", "FIXED", "FESTIVE", "BOLA", "RL", "sim_rl", "sim_dp",
    "bb", "rl", "mpc", "cmc", "bola", "netllm", "quetra", "genet", "ppo",
    "test_ppo",
]


# ----------------------------- data model -----------------------------

@dataclass
class SessionData:
    scheme: str
    session_id: str
    time_s: np.ndarray
    bitrate_kbps: np.ndarray
    buffer_s: np.ndarray
    rebuffer_s: np.ndarray
    bandwidth_mbps: np.ndarray
    reward: np.ndarray

    @property
    def n(self) -> int:
        return len(self.time_s)

    def truncated(self, n: Optional[int]) -> "SessionData":
        if n is None or self.n <= n:
            return self
        sl = slice(0, n)
        return SessionData(
            scheme=self.scheme,
            session_id=self.session_id,
            time_s=self.time_s[sl],
            bitrate_kbps=self.bitrate_kbps[sl],
            buffer_s=self.buffer_s[sl],
            rebuffer_s=self.rebuffer_s[sl],
            bandwidth_mbps=self.bandwidth_mbps[sl],
            reward=self.reward[sl],
        )


@dataclass
class AggregateMetrics:
    scheme: str
    session_ids: List[str]
    total_reward: List[float]
    mean_reward: List[float]
    mean_bitrate_mbps: List[float]
    stall_ratio_pct: List[float]
    smoothness_mbps: List[float]
    total_rebuffer_s: List[float]

    @property
    def n(self) -> int:
        return len(self.session_ids)


# ----------------------------- parsing -----------------------------

def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_time_like(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = arr - arr[0]
    return arr


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def infer_scheme_and_session_id(filename: str, known_schemes: Optional[Sequence[str]] = None) -> Tuple[str, str]:
    """
    Supports:
      log_SCHEME_trace.txt
      arbitrary names containing a known scheme token
    """
    name = Path(filename).name
    if name.startswith("log_"):
        parts = name.split("_")
        if len(parts) >= 3:
            scheme = parts[1]
            session = "_".join(parts[2:])
            return scheme, session

    known = list(known_schemes or DEFAULT_SCHEME_ORDER)
    lower = name.lower()
    matches = [s for s in known if s.lower() in lower]
    if matches:
        # longest match first to avoid rl matching inside sim_rl
        matches.sort(key=len, reverse=True)
        scheme = matches[0]
        session = re.sub(re.escape(scheme), "", name, flags=re.IGNORECASE).strip("_-")
        return scheme, session or name

    return Path(name).stem, Path(name).stem


def _parse_standard_log(lines: Sequence[str], cfg: ABRPlotConfig) -> Tuple[np.ndarray, ...]:
    """
    Expected columns, based on the older scripts:
      0 time
      1 bitrate
      2 buffer
      3 rebuffer
      4 bytes downloaded or throughput numerator
      5 download time denominator
      6 reward
    """
    time_s, bitrate_kbps, buffer_s, rebuffer_s, bandwidth_mbps, reward = [], [], [], [], [], []

    for line in lines:
        sp = line.split()
        if len(sp) <= 1:
            continue
        try:
            t = float(sp[0])
            br = float(sp[1])
            buf = float(sp[2])
            reb = float(sp[3])
            # preserve legacy conversion from plot_results.py / plot_neuralABR.py
            bw = float(sp[4]) / float(sp[5]) * cfg.bits_in_byte * cfg.millisec_in_sec / cfg.million
            rew = float(sp[6])
        except Exception:
            continue

        time_s.append(t)
        bitrate_kbps.append(br)
        buffer_s.append(buf)
        rebuffer_s.append(reb)
        bandwidth_mbps.append(bw)
        reward.append(rew)

    time_s = _normalize_time_like(time_s)
    return (
        np.asarray(time_s, dtype=float),
        np.asarray(bitrate_kbps, dtype=float),
        np.asarray(buffer_s, dtype=float),
        np.asarray(rebuffer_s, dtype=float),
        np.asarray(bandwidth_mbps, dtype=float),
        np.asarray(reward, dtype=float),
    )


def _parse_simdp_log(lines: Sequence[str], cfg: ABRPlotConfig) -> Tuple[np.ndarray, ...]:
    """
    Supports the sim_dp variant found in plot_results.py / plot_neuralABR.py:
      - lines with >= 7 columns contain quality index and timing columns
      - one bare numeric line may encode a session-level reward
    """
    if not cfg.video_bitrates_kbps:
        raise ValueError("sim_dp parsing requires cfg.video_bitrates_kbps")

    time_s, bitrate_kbps, buffer_s, rebuffer_s, bandwidth_mbps = [], [], [], [], []
    scalar_reward: Optional[float] = None

    for line in lines:
        sp = line.split()
        if len(sp) == 1:
            maybe = _safe_float(sp[0], default=float("nan"))
            if not math.isnan(maybe):
                scalar_reward = maybe
            continue

        if len(sp) >= 7:
            try:
                t = float(sp[3])
                buf = float(sp[4])
                bw = float(sp[5])
                q = int(sp[6])
            except Exception:
                continue

            q = max(0, min(q, len(cfg.video_bitrates_kbps) - 1))
            time_s.append(t)
            bitrate_kbps.append(float(cfg.video_bitrates_kbps[q]))
            buffer_s.append(buf)
            bandwidth_mbps.append(bw)
            rebuffer_s.append(0.0)

    if not time_s:
        return tuple(np.asarray([], dtype=float) for _ in range(6))  # type: ignore[return-value]

    # legacy sim_dp logs are reverse chronological in old scripts
    time_s = list(reversed(time_s))
    bitrate_kbps = list(reversed(bitrate_kbps))
    buffer_s = list(reversed(buffer_s))
    bandwidth_mbps = list(reversed(bandwidth_mbps))
    rebuffer_s = list(reversed(rebuffer_s))

    reward = _derive_reward_from_series(
        bitrate_kbps=np.asarray(bitrate_kbps, dtype=float),
        buffer_s=np.asarray(buffer_s, dtype=float),
        time_s=np.asarray(time_s, dtype=float),
        qoe=cfg.qoe,
        session_level_reward=scalar_reward,
    )

    return (
        _normalize_time_like(time_s),
        np.asarray(bitrate_kbps, dtype=float),
        np.asarray(buffer_s, dtype=float),
        np.asarray(rebuffer_s, dtype=float),
        np.asarray(bandwidth_mbps, dtype=float),
        np.asarray(reward, dtype=float),
    )


def _derive_reward_from_series(
    bitrate_kbps: np.ndarray,
    buffer_s: np.ndarray,
    time_s: np.ndarray,
    qoe: QoEConfig,
    session_level_reward: Optional[float] = None,
) -> np.ndarray:
    """
    Reconstruct a per-chunk reward when not explicitly logged.
    Uses the same spirit as the older sim_dp reward reconstruction, but
    keeps units consistent and robust.
    """
    n = len(bitrate_kbps)
    if n == 0:
        return np.asarray([], dtype=float)

    reward = np.zeros(n, dtype=float)
    prev_bitrate = bitrate_kbps[0]
    prev_time = time_s[0]
    prev_buffer = buffer_s[0]

    for i in range(n):
        t = time_s[i]
        b = buffer_s[i]
        br = bitrate_kbps[i]

        # approximate incremental stall from time jump minus prior playable buffer
        stall = max(0.0, (t - prev_time) - prev_buffer) if i > 0 else 0.0
        reward[i] = (
            br / qoe.kbps_per_mbps
            - qoe.rebuffer_penalty * stall
            - qoe.smooth_penalty * abs(br - prev_bitrate) / qoe.kbps_per_mbps
        )
        prev_bitrate = br
        prev_time = t
        prev_buffer = b

    if session_level_reward is not None and n > 1:
        # preserve the session scalar approximately by spreading delta across valid chunks
        current_sum = float(np.sum(reward[1:]))
        delta = session_level_reward - current_sum
        reward[1:] += delta / max(1, n - 1)

    return reward


def parse_log_file(path: Path, cfg: ABRPlotConfig, known_schemes: Optional[Sequence[str]] = None) -> SessionData:
    scheme, session_id = infer_scheme_and_session_id(path.name, known_schemes=known_schemes)
    scheme = cfg.canonical_scheme(scheme)

    lines = _read_lines(path)
    is_simdp = "sim_dp" in path.name.lower() or scheme.lower() == "sim_dp"

    if is_simdp:
        parsed = _parse_simdp_log(lines, cfg)
    else:
        parsed = _parse_standard_log(lines, cfg)

    time_s, bitrate_kbps, buffer_s, rebuffer_s, bandwidth_mbps, reward = parsed
    return SessionData(
        scheme=scheme,
        session_id=session_id,
        time_s=time_s,
        bitrate_kbps=bitrate_kbps,
        buffer_s=buffer_s,
        rebuffer_s=rebuffer_s,
        bandwidth_mbps=bandwidth_mbps,
        reward=reward,
    )


def load_sessions(
    results_dir: Path | str,
    cfg: ABRPlotConfig,
    schemes: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, SessionData]]:
    results_dir = Path(results_dir)
    sessions: Dict[str, Dict[str, SessionData]] = {}
    known_schemes = list(schemes or DEFAULT_SCHEME_ORDER)

    for path in sorted(results_dir.iterdir()):
        if not path.is_file():
            continue
        sess = parse_log_file(path, cfg, known_schemes=known_schemes)
        if sess.n == 0:
            continue
        if schemes and sess.scheme not in schemes:
            continue
        sessions.setdefault(sess.scheme, {})[sess.session_id] = sess
    return sessions


# ----------------------------- alignment + metrics -----------------------------

def common_session_ids(data: Dict[str, Dict[str, SessionData]], schemes: Optional[Sequence[str]] = None, min_len: int = 1) -> List[str]:
    scheme_list = list(schemes or data.keys())
    if not scheme_list:
        return []

    common: Optional[set[str]] = None
    for scheme in scheme_list:
        ids = {sid for sid, s in data.get(scheme, {}).items() if s.n >= min_len}
        common = ids if common is None else common.intersection(ids)
    return sorted(common or [])


def compute_session_metrics(session: SessionData, cfg: ABRPlotConfig) -> Dict[str, float]:
    n = session.n if cfg.video_len is None else min(session.n, cfg.video_len)
    s = session.truncated(n)
    if s.n == 0:
        return {
            "total_reward": 0.0,
            "mean_reward": 0.0,
            "mean_bitrate_mbps": 0.0,
            "smoothness_mbps": 0.0,
            "total_rebuffer_s": 0.0,
            "stall_ratio_pct": 0.0,
        }

    reward_slice = s.reward[1:] if s.n > 1 else s.reward
    total_reward = float(np.sum(reward_slice))
    mean_reward = float(np.mean(reward_slice)) if reward_slice.size else 0.0
    mean_bitrate_mbps = float(np.mean(s.bitrate_kbps) / 1000.0)
    smoothness_mbps = float(np.mean(np.abs(np.diff(s.bitrate_kbps))) / 1000.0) if s.n > 1 else 0.0
    total_rebuffer_s = float(np.sum(s.rebuffer_s[1:])) if s.n > 1 else float(np.sum(s.rebuffer_s))

    # infer content duration from actual sample span when available
    content_time_s = max(float(s.time_s[-1]), 0.0) if s.n > 0 else 0.0
    denom = content_time_s + total_rebuffer_s
    stall_ratio_pct = 100.0 * total_rebuffer_s / denom if denom > 0 else 0.0

    return {
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "mean_bitrate_mbps": mean_bitrate_mbps,
        "smoothness_mbps": smoothness_mbps,
        "total_rebuffer_s": total_rebuffer_s,
        "stall_ratio_pct": stall_ratio_pct,
    }


def aggregate_metrics(
    data: Dict[str, Dict[str, SessionData]],
    cfg: ABRPlotConfig,
    schemes: Optional[Sequence[str]] = None,
    min_len: int = 1,
) -> Dict[str, AggregateMetrics]:
    scheme_list = list(schemes or data.keys())
    aligned_ids = common_session_ids(data, schemes=scheme_list, min_len=min_len)
    out: Dict[str, AggregateMetrics] = {}

    for scheme in scheme_list:
        totals, means, bitrates, stalls, smooths, rebufs = [], [], [], [], [], []
        for sid in aligned_ids:
            s = data[scheme][sid]
            m = compute_session_metrics(s, cfg)
            totals.append(m["total_reward"])
            means.append(m["mean_reward"])
            bitrates.append(m["mean_bitrate_mbps"])
            stalls.append(m["stall_ratio_pct"])
            smooths.append(m["smoothness_mbps"])
            rebufs.append(m["total_rebuffer_s"])
        out[scheme] = AggregateMetrics(
            scheme=scheme,
            session_ids=aligned_ids,
            total_reward=totals,
            mean_reward=means,
            mean_bitrate_mbps=bitrates,
            stall_ratio_pct=stalls,
            smoothness_mbps=smooths,
            total_rebuffer_s=rebufs,
        )
    return out


# ----------------------------- stats helpers -----------------------------

def confidence_interval_95(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    half = 1.96 * se
    return mean, mean - half, mean + half


def cdf_points(samples: Sequence[float], margin: float = 0.025) -> np.ndarray:
    """
    Step-style CDF, inspired by plot_all_sabre.py, but returned as a numpy array.
    """
    if not samples:
        return np.zeros((0, 2), dtype=float)
    s = np.sort(np.asarray(samples, dtype=float))
    span = s[-1] - s[0]
    m = margin * span if span > 0 else margin
    inc = 1.0 / len(s)
    y = 0.0
    pts: List[Tuple[float, float]] = []
    if span == 0:
        pts.append((s[0] - m, y))
    for x in s:
        pts.append((float(x), y))
        y += inc
        pts.append((float(x), y))
    if span == 0:
        pts.append((s[-1] + m, y))
    return np.asarray(pts, dtype=float)


# ----------------------------- plotting helpers -----------------------------

def _finalize(ax: plt.Axes, xlabel: str, ylabel: str, title: Optional[str] = None, legend: bool = True) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(frameon=False)


def save_png(fig: plt.Figure, out_path: Path | str, dpi: int = 180, close: bool = True) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return out_path


# ----------------------------- public plot API -----------------------------

def plot_reward_by_trace(
    metrics: Dict[str, AggregateMetrics],
    out_path: Optional[Path | str] = None,
    title: str = "Total Reward by Trace",
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for scheme, m in metrics.items():
        ax.plot(m.total_reward, marker="o", linewidth=1.8, markersize=4, label=f"{scheme} ({np.mean(m.total_reward):.2f})")
    _finalize(ax, "Trace index", "Total reward", title)
    if out_path:
        save_png(fig, out_path)
    return fig, ax


def plot_qoe_cdf(
    metrics: Dict[str, AggregateMetrics],
    out_path: Optional[Path | str] = None,
    title: str = "QoE CDF",
    use_mean_reward: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for scheme, m in metrics.items():
        samples = m.mean_reward if use_mean_reward else m.total_reward
        pts = cdf_points(samples)
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], linewidth=2, label=f"{scheme} ({np.mean(samples):.2f})")
    _finalize(ax, "QoE", "CDF", title)
    ax.set_ylim(0, 1.01)
    if out_path:
        save_png(fig, out_path)
    return fig, ax


def plot_tradeoff_scatter(
    metrics: Dict[str, AggregateMetrics],
    x: str,
    y: str,
    out_path: Optional[Path | str] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    axis_labels = {
        "stall_ratio_pct": "Time spent on stall (%)",
        "smoothness_mbps": "Bitrate smoothness (Mbps)",
        "mean_bitrate_mbps": "Video bitrate (Mbps)",
        "total_reward": "Total reward",
        "mean_reward": "Mean reward",
        "total_rebuffer_s": "Total rebuffer (s)",
    }

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for scheme, m in metrics.items():
        xv = getattr(m, x)
        yv = getattr(m, y)
        mx, lx, hx = confidence_interval_95(xv)
        my, ly, hy = confidence_interval_95(yv)
        ax.errorbar(
            mx, my,
            xerr=[[mx - lx], [hx - mx]],
            yerr=[[my - ly], [hy - my]],
            marker="o",
            capsize=4,
            linewidth=1.5,
            label=scheme,
        )

    if title is None:
        title = f"{axis_labels.get(y, y)} vs {axis_labels.get(x, x)}"
    _finalize(ax, axis_labels.get(x, x), axis_labels.get(y, y), title)
    if out_path:
        save_png(fig, out_path)
    return fig, ax


def plot_session_panel(
    sessions_by_scheme: Dict[str, Dict[str, SessionData]],
    session_id: str,
    schemes: Optional[Sequence[str]] = None,
    out_path: Optional[Path | str] = None,
    title: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    scheme_list = list(schemes or sessions_by_scheme.keys())
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    fields = [
        ("bitrate_kbps", "Bitrate (kbps)"),
        ("buffer_s", "Buffer (s)"),
        ("bandwidth_mbps", "Bandwidth (Mbps)"),
        ("reward", "Reward"),
    ]

    for scheme in scheme_list:
        s = sessions_by_scheme.get(scheme, {}).get(session_id)
        if s is None:
            continue
        if limit is not None:
            s = s.truncated(limit)
        for ax, (field, ylabel) in zip(axes, fields):
            ax.plot(s.time_s, getattr(s, field), linewidth=1.8, label=scheme)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)")
    if title is None:
        title = f"Session: {session_id}"
    fig.suptitle(title, y=0.995)
    axes[0].legend(frameon=False, ncol=min(4, max(1, len(scheme_list))))
    if out_path:
        save_png(fig, out_path)
    return fig, axes


def plot_overlay_timeseries(
    sessions_by_scheme: Dict[str, Dict[str, SessionData]],
    session_id: str,
    field: str,
    ylabel: str,
    schemes: Optional[Sequence[str]] = None,
    out_path: Optional[Path | str] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    scheme_list = list(schemes or sessions_by_scheme.keys())
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for scheme in scheme_list:
        s = sessions_by_scheme.get(scheme, {}).get(session_id)
        if s is None:
            continue
        ax.plot(s.time_s, getattr(s, field), linewidth=1.8, label=scheme)
    _finalize(ax, "Time (s)", ylabel, title or f"{field} overlay: {session_id}")
    if out_path:
        save_png(fig, out_path)
    return fig, ax


# ----------------------------- convenience runner -----------------------------

def build_default_report(
    results_dir: Path | str,
    output_dir: Path | str,
    cfg: Optional[ABRPlotConfig] = None,
    schemes: Optional[Sequence[str]] = None,
    min_len: int = 2,
) -> Dict[str, Path]:
    cfg = cfg or ABRPlotConfig(results_dir=results_dir, output_dir=output_dir)
    cfg.results_dir = Path(results_dir)
    cfg.output_dir = Path(output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_sessions(cfg.results_dir, cfg, schemes=schemes)
    metrics = aggregate_metrics(data, cfg, schemes=schemes, min_len=min_len)

    files: Dict[str, Path] = {}
    files["reward_by_trace"] = save_png(plot_reward_by_trace(metrics)[0], cfg.output_dir / "reward_by_trace.png")
    files["qoe_cdf"] = save_png(plot_qoe_cdf(metrics)[0], cfg.output_dir / "qoe_cdf.png")
    files["bitrate_vs_stall"] = save_png(
        plot_tradeoff_scatter(metrics, x="stall_ratio_pct", y="mean_bitrate_mbps", title="Average bitrate vs stall ratio")[0],
        cfg.output_dir / "bitrate_vs_stall.png",
    )
    files["smoothness_vs_stall"] = save_png(
        plot_tradeoff_scatter(metrics, x="stall_ratio_pct", y="smoothness_mbps", title="Smoothness vs stall ratio")[0],
        cfg.output_dir / "smoothness_vs_stall.png",
    )
    files["bitrate_vs_smoothness"] = save_png(
        plot_tradeoff_scatter(metrics, x="smoothness_mbps", y="mean_bitrate_mbps", title="Average bitrate vs smoothness")[0],
        cfg.output_dir / "bitrate_vs_smoothness.png",
    )

    ids = common_session_ids(data, schemes=schemes, min_len=min_len)
    if ids:
        files["session_panel"] = save_png(
            plot_session_panel(data, ids[0], schemes=schemes, title=f"Representative session: {ids[0]}")[0],
            cfg.output_dir / "session_panel.png",
        )

    return files
