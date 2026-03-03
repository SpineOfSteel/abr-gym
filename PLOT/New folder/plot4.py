# pip install mpld3
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins

OUT = Path("../_CLIENT_LOGS")
PAT = "log_*_.txt"
LAST_N = None            # None = all
SAVE = True             # writes 4 html fragments into ./output
SHOW_INLINE = False     # True if you're in a notebook and want interactive inline

def algo(p):
    m = re.match(r"^log_(.+)_\.txt$", p.name)
    return m.group(1) if m else p.stem

def read(p):
    rows = []
    for ln in p.read_text(errors="ignore").splitlines():
        sp = ln.split()
        if len(sp) < 7: 
            continue
        try: rows.append([float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]), float(sp[6])])
        except: pass
    if not rows: return None
    a = np.asarray(rows, float)
    t = a[:,0] - a[0,0]
    br, buf, rb, rw = a[:,1], a[:,2], a[:,3], a[:,4]
    if LAST_N is not None and len(t) > LAST_N:
        sl = slice(-LAST_N, None)
        t, br, buf, rb, rw = t[sl], br[sl], buf[sl], rb[sl], rw[sl]
    return t, br, buf, rb, rw


data = {}
for p in sorted(OUT.glob(PAT)):
    d = read(p)
    if d is not None: data[algo(p)] = d
if not data: raise SystemExit(f"No parseable logs in {OUT} matching {PAT}")

def fig(kind, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    lines, labels = [], []
    for a, (t, br, buf, rb, rw) in sorted(data.items()):
        y = {"reward": rw, "bitrate": br, "buffer": buf, "rebuffer": rb}[kind]
        ln = ax.plot(t, y, lw=1)[0]
        lines.append(ln); labels.append(a)
    ax.set_title(title); ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel)
    plugins.connect(fig, plugins.InteractiveLegendPlugin(lines, labels, alpha_unsel=0.15))
    fig.tight_layout()
    return fig

figs = {
    "reward":   fig("reward",   "QoE / Reward",  "Reward"),
    "bitrate":  fig("bitrate",  "Bitrate (Kbps)","Bitrate"),
    "buffer":   fig("buffer",   "Buffer (s)",    "Buffer"),
    "rebuffer": fig("rebuffer", "Rebuffer (s)",  "Rebuffer"),
}

if SAVE:
    parts = []
    for title, f in figs.items():
        parts.append(f"<h2>{title}</h2>\n")
        parts.append(mpld3.fig_to_html(f))
        parts.append("\n<hr/>\n")
        plt.close(f)

if SHOW_INLINE:
    for f in figs.values():
        mpld3.display(f)

# optional: also show static matplotlib windows
# plt.show()
