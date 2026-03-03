
from pathlib import Path
import numpy as np
import os

# Print the directory
print("Current Working Directory:", os.getcwd())

from abr_plot_lib import (
    ABRPlotConfig,
    build_default_report,
    load_sessions,
    aggregate_metrics,
    common_session_ids,
    plot_session_panel,
    plot_overlay_timeseries,
    save_png,
)

ROOT = Path("")
RESULTS = ROOT / "example_results"
FIGS = ROOT / "example_figures"


def make_standard_log(path: Path, bitrate_base: float, bandwidth_base: float, reward_bias: float) -> None:
    lines = []
    t = 1000.0
    buf = 1.5
    for i in range(48):
        br = bitrate_base + 150 * (i % 5)
        reb = 0.18 if i in (0, 7) else 0.0
        bw_num = (bandwidth_base + 0.08 * np.sin(i / 4.0)) * 1_000_000
        bw_den = 1000.0
        rew = br / 1000.0 - 4.3 * reb - abs(150 if i else 0) / 1000.0 + reward_bias
        lines.append(f"{t:.2f} {br:.0f} {buf:.3f} {reb:.3f} {bw_num:.2f} {bw_den:.1f} {rew:.3f}")
        t += 1.0
        buf = max(0.2, min(12.0, buf + 0.7 - reb))
    path.write_text("\n".join(lines), encoding="utf-8")


def make_demo_logs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for trace_i in range(1, 4):
        make_standard_log(RESULTS / f"log_BB_trace{trace_i}.txt", 900 + 30 * trace_i, 2.1, -0.08)
        make_standard_log(RESULTS / f"log_BOLA_trace{trace_i}.txt", 1200 + 45 * trace_i, 2.5, 0.05)
        make_standard_log(RESULTS / f"log_RL_trace{trace_i}.txt", 1450 + 50 * trace_i, 2.8, 0.12)


def main() -> None:
    #make_demo_logs()

    cfg = ABRPlotConfig(
        results_dir=RESULTS,
        output_dir=FIGS,
        video_len=48,
        video_bitrates_kbps=[300, 750, 1200, 1850, 2850, 4300],  # used only if sim_dp logs appear
        scheme_aliases={"bb": "BB", "bola": "BOLA", "rl": "RL"},
    )

    generated = build_default_report(RESULTS, FIGS, cfg=cfg, schemes=["BB", "BOLA", "RL"])
    print("Generated core PNGs:")
    for k, v in generated.items():
        print(f"  {k}: {v}")

    sessions = load_sessions(RESULTS, cfg, schemes=["BB", "BOLA", "RL"])
    metrics = aggregate_metrics(sessions, cfg, schemes=["BB", "BOLA", "RL"], min_len=2)
    ids = common_session_ids(sessions, schemes=["BB", "BOLA", "RL"], min_len=2)

    if ids:
        sid = ids[0]
        save_png(
            plot_overlay_timeseries(
                sessions, sid, field="bitrate_kbps", ylabel="Bitrate (kbps)", schemes=["BB", "BOLA", "RL"]
            )[0],
            FIGS / "overlay_bitrate.png",
        )
        save_png(
            plot_session_panel(
                sessions, sid, schemes=["BB", "BOLA", "RL"], title=f"Detailed session panel: {sid}"
            )[0],
            FIGS / "session_panel_detailed.png",
        )
        print(f"Extra session plots written for {sid}")


if __name__ == "__main__":
    main()
