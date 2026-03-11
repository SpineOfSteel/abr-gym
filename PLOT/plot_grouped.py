import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import scipy.stats
print(os.getcwd())

from typing import Callable, Optional
import pandas as pd

from stable_baselines3.common.monitor import load_results


plt.switch_backend('agg')

NUM_BINS = 500
VIDEO_LEN = 48
LW = 1.5

SOURCE_LOG = '..\\DATASET\\TRACES\\norway'
OUTPUT_DIR = './graphs'
print(Path(SOURCE_LOG).exists())

TRANSPORTS = ['tram', 'car', 'bus','ferry','metro','train']
SCHEMES = ['bb', 'bola', 'mpc',  'rl', 'ppo', 'netllm']
LABELS = ['BBA', 'BOLA', 'RobustMPC','Pensieve',  'Pen-PPO',   'NetLLM']
COLORS = ['#4E79A7', '#F28E2B', '#E15759', "#21EBDA", '#59A14F', "#FA12B4", '#B07AA1', '#FF9DA7', '#9C755F']






#
# LOG FILE PARSING AND METRIC CALCULATION
#
def collect_scheme_metrics(root_dir, transport=None):
    metrics = {scheme: {'reward': [], 'bitrate': [], 'stall': [], 'smoothness': []} for scheme in SCHEMES}
    files = []
    root = Path(root_dir)
    if root.exists():
        files = [path for path in root.rglob('*.txt') if transport is None or f'_{transport}_' in path.stem.lower()]
        #print(files[0].stem.split('_')[-1], files[0])
        files = sorted(files, key=lambda file: int(file.stem.split('_')[-1]))
    
        for path in files:
            for scheme in SCHEMES:
                if scheme in path.stem.lower():
                    summary = summarize_metrics(parse_log_file(path))
                    if summary is None:
                        break
                    metrics[scheme]['reward'].append(summary['mean_reward'])
                    metrics[scheme]['bitrate'].append(summary['mean_bitrate'])
                    metrics[scheme]['stall'].append(summary['stall_pct'])
                    metrics[scheme]['smoothness'].append(summary['smoothness'])
                    break
    return metrics


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



#
# PLOTTING
#
def main():
    group_plots(SOURCE_LOG, transport=None)
    for transport in TRANSPORTS:
        group_plots(SOURCE_LOG, transport=transport)

def group_plots(root_dir, transport=None):
    metrics = collect_scheme_metrics(root_dir, transport=transport)
    suffix = transport if transport else 'all'
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    skip = 3 if transport is None else 1
    plot_tradeoff_scatter(metrics, 'bitrate', 'stall', str(outdir / f'baselines-{suffix}-tradeoff.png'), title=f'{suffix.upper()} (error plot)')
    plot(metrics,'smoothness', 'Traces', 'Bitrate Smoothness', skip, str(outdir / f'baselines-{suffix}-sr.png'))
    plot(metrics,'bitrate', 'Traces', 'Video Bitrate', skip, str(outdir / f'baselines-{suffix}-br.png'))
    plot(metrics,'stall', 'Traces', 'Time Spent on Stall', skip, str(outdir / f'baselines-{suffix}-st.png'))
    #plot(metrics,'reward', 'Traces', 'Reward', skip, str(outdir / f'baselines-{suffix}-rew.png'))
    plot_qoe_cdf(metrics, str(outdir / f'baselines-{suffix}-qoe.png'), title_suffix=suffix.upper())



def plot(vals, metric, xlabel, ylabel, skip, output_path, invert_x=False, invert_y=False, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 5))    
    for idx, scheme in enumerate(SCHEMES):
        ys = vals[scheme][metric]
        avg_y = np.mean(ys)
        

        ys_down = ys[::skip]
        xs = np.arange(len(ys_down))
        
        ax.plot(
            xs, ys_down, 
            color=COLORS[idx], 
            label=f"{LABELS[idx]} (Mean: {avg_y:.2f})", 
            linewidth=1.5,
            antialiased=True,
            zorder=3
        )
    fix_axes(ax, xlabel=xlabel, ylabel=ylabel, invert_x=invert_x, invert_y=invert_y, xlim=xlim, ylim=ylim)
    ax.set_xticks(np.arange(min(xs), max(xs) + 1, 5))
    fig.savefig(output_path)
    plt.close(fig)
   

def plot_qoe_cdf(metrics, output_path, title_suffix=''):
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, scheme in enumerate(SCHEMES):
        rewards = metrics[scheme]['reward']
        values, base = np.histogram(rewards, bins=NUM_BINS)
        cumulative = np.cumsum(values)
        if cumulative[-1] == 0:
            continue
        cumulative = cumulative / cumulative[-1]
        ax.plot(base[:-1], cumulative, '-', color=COLORS[idx], lw=LW,
                label=f'{LABELS[idx]}: {np.mean(rewards):.2f}')
        print(f'{scheme}, {np.mean(rewards):.2f}')

    
    fix_axes(ax, xlabel='QoE', ylabel='CDF', title_suffix=title_suffix)
    ax.set_ylim(0.0, 1.01)
    fig.savefig(output_path)
    plt.close(fig)
    

def plot_tradeoff_scatter(metrics,x,y,output_path, title=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, scheme in enumerate(SCHEMES):
        m = metrics[scheme]
        x_ = m[x]
        y_ = m[y]
        mx, my = np.mean(x_), np.mean(y_)
        lx, hx = np.percentile(x_, [25, 75])
        ly, hy = np.percentile(y_, [25, 75])

        xerr = np.array([[max(0, mx - lx)], [max(0, hx - mx)]])
        yerr = np.array([[max(0, my - ly)], [max(0, hy - my)]])
        #print(xerr, yerr)
        legend_label = f"{LABELS[idx]} ({mx:.2f}, {my:.2f})"
        ax.errorbar(
            mx, my,
            xerr=xerr,
            yerr=yerr,
            fmt='o',
            color=COLORS[idx],
            capsize=4,
            linewidth=1.5,
             label=legend_label,
        )


    
    fix_axes(ax, xlabel='bitrate', ylabel='stall ratio (%)', title_suffix=title)
    ax.set_ylim(0.0, 1.01)
    fig.savefig(output_path)
    plt.close(fig)
    

def fix_axes(ax, xlabel='X', ylabel='Y', title_suffix='', invert_x=False, invert_y=False, xlim=None, ylim=None,):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(linestyle='--', linewidth=1.0, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, ncol=2, edgecolor='white', loc='upper right')
    if title_suffix:
        ax.set_title(title_suffix)
    


X_TIMESTEPS = "timesteps"
X_EPISODES = "episodes"
X_WALLTIME = "walltime_hrs"
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100


def rolling_window(array: np.ndarray, window: int) -> np.ndarray:
    """
    Apply a rolling window to a np.ndarray

    :param array: the input Array
    :param window: length of the rolling window
    :return: rolling window on the input array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)  # noqa: RUF005
    strides = (*array.strides, array.strides[-1])
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def window_func(var_1: np.ndarray, var_2: np.ndarray, window: int, func: Callable) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a function to the rolling window of 2 arrays

    :param var_1: variable 1
    :param var_2: variable 2
    :param window: length of the rolling window
    :param func: function to apply on the rolling window on variable 2 (such as np.mean)
    :return:  the rolling output with applied function
    """
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return var_1[window - 1 :], function_on_var2


def ts2xy(data_frame: pd.DataFrame, x_axis: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose a data frame variable to x and ys
    (y = episodic return)

    :param data_frame: the input data
    :param x_axis: the x-axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: the x and y output
    """
    if x_axis == X_TIMESTEPS:
        x_var = np.cumsum(data_frame.l.values)  # type: ignore[arg-type]
        y_var = data_frame.r.values
    elif x_axis == X_EPISODES:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.r.values
    elif x_axis == X_WALLTIME:
        # Convert to hours
        x_var = data_frame.t.values / 3600.0  # type: ignore[operator, assignment]
        y_var = data_frame.r.values
    else:
        raise NotImplementedError(f"Unsupported {x_axis=}, please use one of {POSSIBLE_X_AXES}")
    return x_var, y_var  # type: ignore[return-value]


def plot_curves(
    xy_list: list[tuple[np.ndarray, np.ndarray]], x_axis: str, title: str, figsize: tuple[int, int] = (8, 2)
) -> None:
    """
    plot the curves

    :param xy_list: the x and y coordinates to plot
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: the title of the plot
    :param figsize: Size of the figure (width, height)
    """

    plt.figure(title, figsize=figsize)
    max_x = max(xy[0][-1] for xy in xy_list)
    min_x = 0
    for _, (x, y) in enumerate(xy_list):
        plt.plot(x, y, label=f'Curve {_}') # Changed to plt.plot
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean)
    plt.xlim(min_x, max_x)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()


def plot_results(
    dirs: list[str], num_timesteps: Optional[int], x_axis: str, task_name: str, figsize: tuple[int, int] = (8, 2)
) -> None:
    """
    Plot the results using csv files from ``Monitor`` wrapper.

    :param dirs: the save location of the results to plot
    :param num_timesteps: only plot the points below this value
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: the title of the task to plot
    :param figsize: Size of the figure (width, height)
    """

    data_frames = []
    for folder in dirs:
        data_frame = load_results(folder)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    plot_curves(xy_list, x_axis, task_name, figsize)


if __name__ == '__main__':
    main()
