import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats
print(os.getcwd())


plt.switch_backend('agg')

NUM_BINS = 500
VIDEO_LEN = 48
LW = 1.5

SOURCE_LOG = '..\\DATASET\\TRACES\\norway_tram'
OUTPUT_DIR = './graphs'
print(Path(SOURCE_LOG).exists())

TRANSPORTS = ['tram', 'car', 'bus']
SCHEMES = ['bb', 'rl', 'mpc', 'cmc', 'bola', 'netllm', 'quetra', 'genet', 'ppo']
LABELS = ['BBA', 'Pensieve', 'RobustMPC', 'Comyco', 'BOLA', 'NetLLM', 'QUETRA', 'Genet', 'Pen-PPO']
MARKERS = ['o', 'x', 'v', '^', '>', '<', 's', 'p', '*']
COLORS = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F']


def mean_confidence_interval(data, confidence=0.95):
    a = np.asarray(data, dtype=float)
    n = len(a)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        return a[0], a[0], a[0]
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def detect_transport(path):
    parts = [p.lower() for p in Path(path).parts]
    for transport in TRANSPORTS:
        if transport in parts:
            return transport
    return None


def gather_files(root_dir, transport=None):
    root = Path(root_dir)
    if not root.exists():
        return []
    files = []
    for path in root.rglob('*.txt'):
        if transport is None or detect_transport(path) == transport:
            files.append(path)
    return sorted(files)


def parse_log_file(path):
    bitrate, rebuffer, reward, time_all = [], [], [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            sp = line.split()
            if len(sp) > 1:
                try:
                    time_all.append(float(sp[0]))
                    bitrate.append(float(sp[1]) / 1000.0)  # kbps -> Mbps
                    rebuffer.append(float(sp[3]))
                    reward.append(float(sp[-1]))
                except (ValueError, IndexError):
                    continue
    return {
        'time': time_all,
        'bitrate': bitrate,
        'rebuffer': rebuffer,
        'reward': reward,
    }


def summarize_metrics(parsed):
    bitrate = parsed['bitrate']
    rebuffer = parsed['rebuffer']
    reward = parsed['reward']
    if len(bitrate) == 0 or len(reward) <= 1:
        return None
    return {
        'mean_reward': float(np.mean(reward[1:])),
        'mean_bitrate': float(np.mean(bitrate)),
        'stall_pct': float(np.sum(rebuffer[1:]) / (VIDEO_LEN * 4.0 + np.sum(rebuffer[1:])) * 100.0),
        'smoothness': float(np.mean(np.abs(np.diff(bitrate)))) if len(bitrate) > 1 else 0.0,
    }


def scheme_matches_file(scheme, path):
    name = path.name.lower()
    stem = path.stem.lower()
    return scheme in name or scheme in stem


def collect_scheme_metrics(root_dir, transport=None):
    metrics = {scheme: {'reward': [], 'bitrate': [], 'stall': [], 'smoothness': []} for scheme in SCHEMES}
    for path in gather_files(root_dir, transport=transport):
        for scheme in SCHEMES:
            if scheme_matches_file(scheme, path):
                summary = summarize_metrics(parse_log_file(path))
                if summary is None:
                    break
                metrics[scheme]['reward'].append(summary['mean_reward'])
                metrics[scheme]['bitrate'].append(summary['mean_bitrate'])
                metrics[scheme]['stall'].append(summary['stall_pct'])
                metrics[scheme]['smoothness'].append(summary['smoothness'])
                break
    return metrics


def style_axes(ax):
    ax.grid(linestyle='--', linewidth=1.0, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_errorbar(xvals, yvals, xlabel, ylabel, output_path, invert_x=False, invert_y=False, xlim=None, ylim=None):
    plt.rcParams['axes.labelsize'] = 15
    matplotlib.rc('font', size=15)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    y_high_max = None
    for idx, scheme in enumerate(SCHEMES):
        xs = xvals[scheme]
        ys = yvals[scheme]
        if not xs or not ys:
            continue
        x_mean, x_low, x_high = mean_confidence_interval(xs)
        y_mean, y_low, y_high = mean_confidence_interval(ys)
        y_high_max = y_high if y_high_max is None else max(y_high_max, y_high)
        ax.errorbar(
            x_mean, y_mean,
            xerr=x_high - x_mean,
            yerr=y_high - y_mean,
            color=COLORS[idx], marker=MARKERS[idx], markersize=10,
            label=LABELS[idx], capsize=4
        )
        print(f'{scheme} {y_mean:.2f} {y_low:.2f} {y_high:.2f} {x_mean:.2f} {x_low:.2f} {x_high:.2f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif y_high_max is not None:
        ax.set_ylim(max(0, y_high_max * 0.5), y_high_max * 1.01)
    if xlim is not None:
        ax.set_xlim(*xlim)
    style_axes(ax)
    ax.legend(fontsize=12, ncol=3, edgecolor='white', loc='best')
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    fig.savefig(output_path)
    plt.close(fig)


def plot_qoe_cdf(metrics, output_path, title_suffix=''):
    plt.rcParams['axes.labelsize'] = 15
    matplotlib.rc('font', size=15)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    plt.subplots_adjust(left=0.06, bottom=0.16, right=0.96, top=0.96)

    for idx, scheme in enumerate(SCHEMES):
        rewards = metrics[scheme]['reward']
        if not rewards:
            continue
        values, base = np.histogram(rewards, bins=NUM_BINS)
        cumulative = np.cumsum(values)
        if cumulative[-1] == 0:
            continue
        cumulative = cumulative / cumulative[-1]
        ax.plot(base[:-1], cumulative, '-', color=COLORS[idx], lw=LW,
                label=f'{LABELS[idx]}: {np.mean(rewards):.2f}')
        print(f'{scheme}, {np.mean(rewards):.2f}')

    ax.set_xlabel('QoE')
    ax.set_ylabel('CDF')
    ax.set_ylim(0.0, 1.01)
    style_axes(ax)
    ax.legend(fontsize=12, ncol=3, edgecolor='white', loc='lower right')
    if title_suffix:
        ax.set_title(title_suffix)
    fig.savefig(output_path)
    plt.close(fig)


def make_plots_for_group(root_dir, transport=None):
    metrics = collect_scheme_metrics(root_dir, transport=transport)
    suffix = transport if transport else 'all'
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_errorbar(
        {k: v['stall'] for k, v in metrics.items()},
        {k: v['bitrate'] for k, v in metrics.items()},
        'Time Spent on Stall (%)', 'Video Bitrate (mbps)',
        str(outdir / f'baselines-{suffix}-br.png'), invert_x=True
    )
    plot_errorbar(
        {k: v['stall'] for k, v in metrics.items()},
        {k: v['smoothness'] for k, v in metrics.items()},
        'Time Spent on Stall (%)', 'Bitrate Smoothness (mbps)',
        str(outdir / f'baselines-{suffix}-sr.png'), invert_x=True, invert_y=True
    )
    plot_errorbar(
        {k: v['smoothness'] for k, v in metrics.items()},
        {k: v['bitrate'] for k, v in metrics.items()},
        'Bitrate Smoothness (mbps)', 'Video Bitrate (mbps)',
        str(outdir / f'baselines-{suffix}-bs.png'), invert_x=True
    )
    plot_qoe_cdf(metrics, str(outdir / f'baselines-{suffix}-qoe.png'), title_suffix=suffix.upper())


def main():
    make_plots_for_group(SOURCE_LOG, transport=None)
    for transport in TRANSPORTS:
        make_plots_for_group(SOURCE_LOG, transport=transport)


if __name__ == '__main__':
    main()
