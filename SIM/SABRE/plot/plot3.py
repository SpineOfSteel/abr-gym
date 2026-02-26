# pip install mpld3
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins

OUT = Path("../_CLIENT_LOGS")
PAT = "log_*_.txt"
LAST_N = 300   # set None to plot all samples
HTML = OUT / "abr_report.html"

def read_log(p: Path):
    rows = []
    for ln in p.read_text(errors="ignore").splitlines():
        sp = ln.split()
        if len(sp) < 7: 
            continue
        try:
            # t, bitrate, buffer, rebuf, reward
            rows.append([float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]), float(sp[6])])
        except:
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
    m = re.match(r"^log_(.+)_\.txt$", p.name)
    return m.group(1) if m else p.stem

files = sorted(OUT.glob(PAT))
data = {algo_name(p): read_log(p) for p in files}
data = {k: v for k, v in data.items() if v is not None}
if not data:
    raise SystemExit(f"No parseable logs in {OUT} matching {PAT}")

def overlay(kind, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    lines, labels = [], []
    for algo, (t, br, buf, rb, rew) in sorted(data.items()):
        y = {"reward": rew, "bitrate": br, "buffer": buf, "rebuffer": rb}[kind]
        ln = ax.plot(t, y, lw=1, label=algo)[0]
        lines.append(ln); labels.append(algo)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    plugins.connect(fig, plugins.InteractiveLegendPlugin(lines, labels, alpha_unsel=0.15))
    return fig

# summary table
def stats_row(algo, d):
    _, br, buf, rb, rew = d
    return (algo, len(rew), float(np.mean(rew)), float(np.mean(br)), float(np.mean(buf)), float(np.sum(rb)))

rows = [stats_row(a, d) for a, d in sorted(data.items())]

parts = ["""<!doctype html><html><head><meta charset="utf-8"/>
<title>ABR Report</title>
<style>
body{font-family:Arial,sans-serif;margin:20px}
table{border-collapse:collapse;width:100%;margin:12px 0 18px}
th,td{border:1px solid #ddd;padding:6px;text-align:left}
th{background:#f6f6f6}
.section{margin:18px 0}
.small{color:#666;font-size:13px}
</style></head><body>
<h1>ABR Report</h1>
<h2>Summary</h2>
<table><thead><tr>
<th>algo</th><th>n</th><th>avg_reward</th><th>avg_bitrate_kbps</th><th>avg_buffer_s</th><th>total_rebuf_s</th>
</tr></thead><tbody>"""]

for algo, n, ar, ab, aBuf, tr in rows:
    parts.append(f"<tr><td>{algo}</td><td>{n}</td><td>{ar:.3f}</td><td>{ab:.1f}</td><td>{aBuf:.3f}</td><td>{tr:.3f}</td></tr>")

parts.append("</tbody></table>")

for kind, yl, tt in [
    ("reward",   "QoE / Reward",  "Reward (toggle algos in legend)"),
    ("bitrate",  "Bitrate (Kbps)","Bitrate"),
    ("buffer",   "Buffer (s)",    "Buffer"),
    ("rebuffer", "Rebuffer (s)",  "Rebuffer"),
]:
    fig = overlay(kind, yl, tt)
    parts.append(f'<div class="section"><h2>{tt}</h2>')
    parts.append(mpld3.fig_to_html(fig))
    parts.append("</div>")
    plt.close(fig)

parts.append("</body></html>")
OUT.mkdir(parents=True, exist_ok=True)
HTML.write_text("".join(parts), encoding="utf-8")
print(f"Wrote: {HTML.resolve()}")
