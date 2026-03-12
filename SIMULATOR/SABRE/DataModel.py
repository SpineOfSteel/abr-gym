
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import math, json
import pandas as pd

def from_parquet(parquet_file_path, index = 0):
    try:
        df = pd.read_parquet(parquet_file_path)
        print(f"Successfully read {parquet_file_path}.")
        json_trace_name = df.index[index]
        json_data = df.loc[json_trace_name]

        json_ = []
        for i in json_data.to_list():
            if i is not None:
                json_.append(i)

        print(f"  Parsed JSON data: { json_[0] }...{json_[-1]}")
        return json_
    except FileNotFoundError:
        print(f"Error: Parquet file not found at {parquet_file_path}. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None
    

@dataclass(frozen=True)
class Manifest:
    segment_time_ms: float
    bitrates_kbps: List[float]
    utilities: List[float]
    segments_bits: List[List[int]]  # segments_bits[seg_idx][q] -> bits

    @staticmethod
    def from_json(path: str, movie_length_s: Optional[float] = None) -> "Manifest":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))

        seg_time = float(obj["segment_duration_ms"])
        if seg_time <= 0:
            raise ValueError(f"Invalid segment_duration_ms={seg_time} in {path}")

        bitrates = list(map(float, obj["bitrates_kbps"]))
        if not bitrates or any(b <= 0 for b in bitrates):
            raise ValueError(f"Invalid bitrates_kbps in {path}: {bitrates}")

        u0 = -math.log(bitrates[0])
        utilities = [math.log(b) + u0 for b in bitrates]

        segs = obj["segment_sizes_bits"]  # list[list[int]]
        if not segs:
            raise ValueError(f"No segment_sizes_bits in {path}")

        if movie_length_s is not None:
            l1 = len(segs)
            l2 = math.ceil(movie_length_s * 1000 / seg_time)
            rep = math.ceil(l2 / l1)
            segs = (segs * rep)[:l2]

        return Manifest(segment_time_ms=seg_time, bitrates_kbps=bitrates, utilities=utilities, segments_bits=segs)


@dataclass(frozen=True)
class NetworkPeriod:
    duration_ms: float
    bandwidth_kbps: float
    latency_ms: float

    @staticmethod
    def from_json(path: str, index:int = 0, multiplier: float = 1.0) -> List["NetworkPeriod"]:
        #obj = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = from_parquet(path, index)
        out: List[NetworkPeriod] = []
        for p in obj:
            dur = float(p["duration_ms"])
            if dur <= 0:
                # Drop invalid periods so the simulator always advances time.
                continue

            bw = float(p["bandwidth_kbps"]) * multiplier
            lat = float(p["latency_ms"])

            if bw < 0 or lat < 0:
                raise ValueError(f"Invalid network period (negative bw/lat): {p}")

            out.append(NetworkPeriod(duration_ms=dur, bandwidth_kbps=bw, latency_ms=lat))

        if not out:
            raise ValueError(
                f"Network trace '{path}' became empty after filtering duration_ms<=0. "
                f"Fix the trace (duration_ms must be > 0)."
            )
        return out


@dataclass(frozen=True)
class DownloadProgress:
    index: int
    quality: int
    size_bits: float
    downloaded_bits: float
    time_ms: float
    time_to_first_bit_ms: float
    abandon_to_quality: Optional[int] = None


@dataclass
class Buffer:
    """Stores qualities; fcc_ms is 'first chunk consumed' inside first segment."""
    manifest: Manifest
    contents: List[int] = field(default_factory=list)
    fcc_ms: float = 0.0

    def level_ms(self) -> float:
        return self.manifest.segment_time_ms * len(self.contents) - self.fcc_ms

    def clear(self) -> None:
        self.contents.clear()
        self.fcc_ms = 0.0


@dataclass
class ReactionTracker:
    """Tracks 'reaction time' to upward sustainable network quality."""
    max_buffer_ms: float
    pending: List[List[float]] = field(default_factory=list)  # [time_ms, target_q, (optional) reached_time_ms]
    total_reaction_ms: float = 0.0

    def _process_matured(self, now_ms: float) -> None:
        cutoff = now_ms - self.max_buffer_ms
        while self.pending and self.pending[0][0] < cutoff:
            p = self.pending.pop(0)
            reaction = self.max_buffer_ms if len(p) == 2 else min(self.max_buffer_ms, p[2] - p[0])
            self.total_reaction_ms += reaction

    def on_network_quality_change(self, now_ms: float, new_q: int, prev_q: Optional[int], buffer_contents: List[int]) -> None:
        self._process_matured(now_ms)
        if prev_q is None or new_q <= prev_q:
            return

        # mark any pending switch-up done if quality drops below its target
        for p in self.pending:
            if len(p) == 2 and int(p[1]) > new_q:
                p.append(now_ms)

        # ignore if buffer already has >= new_q, or if already pending
        if any(new_q <= q for q in buffer_contents):
            return
        if any(len(p) >= 2 and new_q <= int(p[1]) for p in self.pending):
            return

        self.pending.append([now_ms, float(new_q)])

    def on_played_quality(self, now_ms: float, q: int) -> None:
        for p in self.pending:
            if len(p) == 2 and q >= int(p[1]):
                p.append(now_ms)

    def on_time_advanced(self, now_ms: float) -> None:
        self._process_matured(now_ms)


@dataclass
class SimStats:
    rebuffer_event_count: int = 0
    rebuffer_time_ms: float = 0.0
    played_utility: float = 0.0
    played_bitrate: float = 0.0
    total_play_time_ms: float = 0.0
    total_bitrate_change: float = 0.0
    total_log_bitrate_change: float = 0.0
    last_played: Optional[int] = None

    overestimate_count: int = 0
    overestimate_avg: float = 0.0
    goodestimate_count: int = 0
    goodestimate_avg: float = 0.0
    estimate_avg: float = 0.0

    rampup_origin_ms: float = 0.0
    rampup_time_ms: Optional[float] = None


@dataclass
class AbrContext:
    manifest: Manifest
    buffer: Buffer
    max_buffer_ms: float
    throughput_est: Optional[float]  # bits/ms
    latency_est: Optional[float]     # ms
    sustainable_quality: Optional[int]
    rampup_threshold: Optional[int]
    stats: SimStats
