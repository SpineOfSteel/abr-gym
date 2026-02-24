#!/usr/bin/env python3

"""

Convert a folder of [timestamp(s) throughput(Mbps)] traces into SABRE network.json format.

Input file format (whitespace-separated):
  <timestamp_seconds> <throughput_mbps>

Output:
  For each input file "foo.txt" -> "foo.json" containing:
    [
      {"duration_ms": <delta_t_ms>, "bandwidth_kbps": <throughput_kbps>, "latency_ms": <latency_ms>},
      ...
    ]
example: 
python .\trace2json.py "C:\Users\raovi\Downloads\full\artifact\Video-Streaming\Network-Traces\Lumous5G\5G" --out-dir "5Glogs" --latency-ms 5 
input: C:\Users\raovi\Downloads\full\artifact\Video-Streaming\Network-Traces\Lumous5G\4G

Notes:
- Throughput(Mbps) -> bandwidth_kbps = Mbps * 1000 (decimal kbps)
- duration_ms is computed from timestamp deltas; last sample uses default_duration_ms
- Optionally prepends a short "zero-bw" warmup period to mimic SABRE examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def parse_trace_lines(text: str) -> List[Tuple[float, float]]:
    rows: List[Tuple[float, float]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 2:
            continue
        try:
            ts = float(parts[0])
            mbps = float(parts[1])
        except ValueError:
            continue
        rows.append((ts, mbps))
    rows.sort(key=lambda x: x[0])
    return rows


def to_sabre_periods(
    rows: List[Tuple[float, float]],
    latency_ms: float,
    default_duration_ms: int,
    min_duration_ms: int,
    prepend_zero: bool,
    prepend_zero_ms: int,
) -> List[dict]:
    if not rows:
        return []

    out: List[dict] = []

    if prepend_zero and prepend_zero_ms > 0:
        out.append({"duration_ms": int(prepend_zero_ms), "bandwidth_kbps": 0, "latency_ms": float(latency_ms)})

    for i, (ts, mbps) in enumerate(rows):
        if i + 1 < len(rows):
            dt_s = rows[i + 1][0] - ts
            dt_ms = int(round(dt_s * 1000.0))
        else:
            dt_ms = int(default_duration_ms)

        # keep SABRE progressing even if timestamps are weird
        if dt_ms <= 0:
            dt_ms = int(min_duration_ms)

        kbps = int(round(mbps * 1000.0))  # Mbps -> kbps (decimal)
        if kbps < 0:
            kbps = 0

        out.append({"duration_ms": dt_ms, "bandwidth_kbps": kbps, "latency_ms": float(latency_ms)})

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", help="Folder containing trace files (txt/trace/etc.)")
    ap.add_argument("--out-dir", default=None, help="Output folder (default: <input_dir>/sabre_json)")
    ap.add_argument("--glob", default="*", help="Which files to read (default: '*')")
    ap.add_argument("--latency-ms", type=float, default=20.0, help="Latency to write into each period")
    ap.add_argument("--default-duration-ms", type=int, default=1000, help="Duration for last sample")
    ap.add_argument("--min-induration-ms", type=int, default=1, help="Clamp for non-positive dt")
    ap.add_argument("--prepend-zero", action="store_true", help="Prepend a short 0-bw warmup period")
    ap.add_argument("--prepend-zero-ms", type=int, default=81, help="Warmup period duration (ms)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Not a directory: {in_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "sabre_json")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob(args.glob) if p.is_file()])
    if not files:
        raise SystemExit(f"No files matched glob '{args.glob}' in {in_dir}")

    converted = 0
    skipped = 0

    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            rows = parse_trace_lines(text)
            if not rows:
                skipped += 1
                continue

            periods = to_sabre_periods(
                rows=rows,
                latency_ms=args.latency_ms,
                default_duration_ms=args.default_duration_ms,
                min_duration_ms=args.min_duration_ms,
                prepend_zero=args.prepend_zero,
                prepend_zero_ms=args.prepend_zero_ms,
            )

            out_path = out_dir / (fp.stem + ".json")
            out_path.write_text(json.dumps(periods, indent=2), encoding="utf-8")
            converted += 1
        except Exception as e:
            print(f"[WARN] Failed {fp.name}: {e}")
            skipped += 1

    print(f"Done. Converted={converted}, Skipped/Failed={skipped}, OutputDir={out_dir}")


if __name__ == "__main__":
    main()
