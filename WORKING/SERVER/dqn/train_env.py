import json
import numpy as np
from typing import Dict

def load_movie_json(path: str, debug: bool = False) -> Dict:
    """
    Trace JSON format (your file):
    [
      {"duration_ms": 1000, "bandwidth_kbps": 45000, "latency_ms": 20.0},
      ...
    ]

    Returns a mahimahi-compatible-ish trace bundle:
      - all_cooked_time: indices [0..N-1]  (compat only)
      - all_cooked_bw: bandwidth_kbps values
      - all_cooked_dur_ms: duration per slot (ms)
      - all_cooked_latency_ms: latency per slot (ms)
    """
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)

    if not isinstance(arr, list) or not arr:
        raise ValueError("Trace JSON must be a non-empty array of objects")

    durations_ms = []
    bw_kbps = []
    latency_ms = []

    for i, row in enumerate(arr):
        if not all(k in row for k in ("duration_ms", "bandwidth_kbps", "latency_ms")):
            raise ValueError(f"Trace entry {i} missing one of: duration_ms, bandwidth_kbps, latency_ms")

        d = float(row["duration_ms"])
        b = float(row["bandwidth_kbps"])
        l = float(row["latency_ms"])

        if d <= 0:
            raise ValueError(f"Trace entry {i} has non-positive duration_ms={d}")
        if b < 0:
            raise ValueError(f"Trace entry {i} has non-positive bandwidth_kbps={b}")
        if l < 0:
            raise ValueError(f"Trace entry {i} has negative latency_ms={l}")

        durations_ms.append(d)
        bw_kbps.append(b)
        latency_ms.append(l)

    durations_ms = np.asarray(durations_ms, dtype=np.float64)
    bw_kbps = np.asarray(bw_kbps, dtype=np.float64)
    latency_ms = np.asarray(latency_ms, dtype=np.float64)

    # As requested: time is just index
    time_idx = np.arange(len(arr), dtype=np.int64)

    out = {
        "trace_id": path,
        "all_cooked_time": [time_idx],              # compatibility
        "all_cooked_bw": [bw_kbps],                 # kbps
        "all_cooked_dur_ms": [durations_ms],        # ms
        "all_cooked_latency_ms": [latency_ms],      # ms
    }

    if debug:
        print(f"[TRACE] Loaded {path}")
        print(f"[TRACE] slots={len(arr)}")
        print(f"[TRACE] bw_kbps first5={bw_kbps[:5].tolist()}")
        print(f"[TRACE] dur_ms first5={durations_ms[:5].tolist()}")
        print(f"[TRACE] lat_ms first5={latency_ms[:5].tolist()}")

    return out



import numpy as np

M_IN_K = 1000.0
MS_IN_S = 1000.0
BITS_IN_BYTE = 8.0

PACKET_PAYLOAD_PORTION = 0.95
BUFFER_THRESH_MS = 60_000.0
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
DRAIN_BUFFER_SLEEP_TIME_MS = 500.0

class Environment:
    """
    Network simulator using trace arrays:
      - all_cooked_time: index array (compat only)
      - all_cooked_bw: bandwidth_kbps
      - all_cooked_dur_ms: slot duration in ms
      - all_cooked_latency_ms: slot latency in ms
    """

    def __init__(
        self,
        all_cooked_time,
        all_cooked_bw,
        all_cooked_dur_ms,
        all_cooked_latency_ms,
        video_size_by_bitrate,        # dict[int] -> list[int bytes]
        chunk_len_ms: float,
        random_seed: int = 42,
        queue_delay_ms: float = 0.0,  # optional extra queueing delay
    ):
        assert len(all_cooked_time) == len(all_cooked_bw) == len(all_cooked_dur_ms) == len(all_cooked_latency_ms)
        assert len(all_cooked_time) > 0

        self.rng = np.random.RandomState(random_seed)

        self.all_cooked_time = all_cooked_time          # compatibility only
        self.all_cooked_bw = all_cooked_bw              # kbps
        self.all_cooked_dur_ms = all_cooked_dur_ms      # ms
        self.all_cooked_latency_ms = all_cooked_latency_ms  # ms

        self.video_size = video_size_by_bitrate         # {q: [bytes per chunk]}
        self.bitrate_levels = len(video_size_by_bitrate)
        self.total_video_chunks = len(next(iter(video_size_by_bitrate.values())))
        self.chunk_len_ms = float(chunk_len_ms)

        self.queue_delay_ms = float(queue_delay_ms)

        self.buffer_size_ms = 0.0
        self.video_chunk_counter = 0

        self.trace_idx = 0
        self.cooked_bw = None
        self.cooked_dur_ms = None
        self.cooked_latency_ms = None

        self.slot_ptr = 0
        self.slot_offset_ms = 0.0  # progress inside current slot

        self._reset_trace()

    def _reset_trace(self):
        self.trace_idx = int(self.rng.randint(len(self.all_cooked_bw)))
        self.cooked_bw = np.asarray(self.all_cooked_bw[self.trace_idx], dtype=np.float64)            # kbps
        self.cooked_dur_ms = np.asarray(self.all_cooked_dur_ms[self.trace_idx], dtype=np.float64)    # ms
        self.cooked_latency_ms = np.asarray(self.all_cooked_latency_ms[self.trace_idx], dtype=np.float64)  # ms

        if len(self.cooked_bw) == 0:
            raise ValueError("Trace has zero slots")

        self.slot_ptr = int(self.rng.randint(len(self.cooked_bw)))
        self.slot_offset_ms = 0.0

    def _advance_slot(self):
        self.slot_ptr += 1
        if self.slot_ptr >= len(self.cooked_bw):
            self.slot_ptr = 0
        self.slot_offset_ms = 0.0

    def _advance_time_without_download(self, sleep_ms: float):
        """Used when draining buffer above threshold (does not add delay)."""
        remain = float(sleep_ms)
        while remain > 0:
            slot_remain_ms = self.cooked_dur_ms[self.slot_ptr] - self.slot_offset_ms
            if slot_remain_ms > remain:
                self.slot_offset_ms += remain
                break
            remain -= slot_remain_ms
            self._advance_slot()

    def reset_episode(self):
        self.buffer_size_ms = 0.0
        self.video_chunk_counter = 0
        self._reset_trace()

    def get_video_chunk(self, quality: int):
        q = int(quality)
        if not (0 <= q < self.bitrate_levels):
            raise ValueError(f"quality must be in [0, {self.bitrate_levels-1}]")

        chunk_size_bytes = float(self.video_size[q][self.video_chunk_counter])

        # Download chunk across trace slots
        delay_ms = 0.0
        bytes_sent = 0.0

        while True:
            bw_kbps = self.cooked_bw[self.slot_ptr]
            slot_remain_ms = self.cooked_dur_ms[self.slot_ptr] - self.slot_offset_ms
            if slot_remain_ms <= 0:
                self._advance_slot()
                continue

            # kbps -> bytes/sec
            throughput_Bps = (bw_kbps * 1000.0) / BITS_IN_BYTE
            slot_remain_s = slot_remain_ms / MS_IN_S

            payload_bytes = throughput_Bps * slot_remain_s * PACKET_PAYLOAD_PORTION

            if bytes_sent + payload_bytes >= chunk_size_bytes:
                remain_bytes = chunk_size_bytes - bytes_sent
                frac_s = remain_bytes / (throughput_Bps * PACKET_PAYLOAD_PORTION)
                frac_ms = frac_s * MS_IN_S

                delay_ms += frac_ms
                self.slot_offset_ms += frac_ms
                break

            bytes_sent += payload_bytes
            delay_ms += slot_remain_ms
            self._advance_slot()

        # Add latency for current slot + optional extra queueing delay
        delay_ms += float(self.cooked_latency_ms[self.slot_ptr]) + self.queue_delay_ms

        # Optional multiplicative noise
        delay_ms *= float(self.rng.uniform(NOISE_LOW, NOISE_HIGH))

        # Rebuffer
        rebuf_ms = max(delay_ms - self.buffer_size_ms, 0.0)

        # Buffer update
        self.buffer_size_ms = max(self.buffer_size_ms - delay_ms, 0.0)
        self.buffer_size_ms += self.chunk_len_ms

        # Buffer cap drain
        sleep_ms = 0.0
        if self.buffer_size_ms > BUFFER_THRESH_MS:
            drain_ms = self.buffer_size_ms - BUFFER_THRESH_MS
            sleep_ms = float(np.ceil(drain_ms / DRAIN_BUFFER_SLEEP_TIME_MS) * DRAIN_BUFFER_SLEEP_TIME_MS)
            self.buffer_size_ms -= sleep_ms
            self._advance_time_without_download(sleep_ms)

        return_buffer_s = self.buffer_size_ms / MS_IN_S

        # Advance video chunk index
        self.video_chunk_counter += 1
        video_chunk_remain = self.total_video_chunks - self.video_chunk_counter
        end_of_video = self.video_chunk_counter >= self.total_video_chunks

        if end_of_video:
            self.reset_episode()

        next_sizes = [self.video_size[i][self.video_chunk_counter] for i in range(self.bitrate_levels)]

        return (
            delay_ms,                    # ms
            sleep_ms,                    # ms
            return_buffer_s,             # s
            rebuf_ms / MS_IN_S,          # s
            int(chunk_size_bytes),       # bytes
            next_sizes,                  # bytes for next chunk at each bitrate
            end_of_video,
            int(video_chunk_remain),
        )


# ABREnv updated (uses trace JSON from load_movie_json and minimizes globals)

import numpy as np

# Keep only core constants global
M_IN_K = 1000.0
S_INFO = 6
S_LEN = 8
BUFFER_NORM_FACTOR = 10.0
DEFAULT_QUALITY = 1
RANDOM_SEED = 42


class ABREnv:
    """
    PPO training env wrapper.
    Assumes load_movie_json(trace_json_path) returns:
      {
        "all_cooked_time": [np.arange(N)],
        "all_cooked_bw": [bandwidth_kbps_array],
        "all_cooked_dur_ms": [duration_ms_array],
        "all_cooked_latency_ms": [latency_ms_array],
      }

    Since trace JSON doesn't include video segment sizes, this builds CBR-like chunk sizes
    from `video_bitrates_kbps` and inferred chunk duration.
    """

    def __init__(
        self,
        trace_json_path,
        video_path,
        random_seed=RANDOM_SEED,
        default_quality=DEFAULT_QUALITY,
        rebuf_penalty=4.3,
        smooth_penalty=1.0,
        queue_delay_ms=0.0,
        debug=False,
    ):
        self.debug = bool(debug)
        self.random_seed = int(random_seed)
        self.default_quality = int(default_quality)
        self.rebuf_penalty = float(rebuf_penalty)
        self.smooth_penalty = float(smooth_penalty)

        # --- load trace json (your "movie.json" is actually a network trace) ---
        trace = load_movie_json(trace_json_path, debug=debug)
        
        self.all_cooked_time = trace["all_cooked_time"]              # index array (compat)
        self.all_cooked_bw = trace["all_cooked_bw"]                  # kbps
        self.all_cooked_dur_ms = trace["all_cooked_dur_ms"]          # ms
        self.all_cooked_latency_ms = trace["all_cooked_latency_ms"]  # ms

        # infer chunk duration from trace slots (assumption: one chunk per trace slot)
        dur0 = np.asarray(self.all_cooked_dur_ms[0], dtype=np.float64)
        if dur0.size == 0:
            raise ValueError("Empty trace durations")
        self.chunk_len_ms = float(np.median(dur0))  # robust if small jitter exists

        # bitrate ladder from caller (not global)
        with open(video_path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        self.video_bitrates = np.asarray(arr['bitrates_kbps'], dtype=np.float32)
        self.a_dim = int(self.video_bitrates.size)
        if self.a_dim <= 0:
            raise ValueError("video_bitrates_kbps must be non-empty")
        if not (0 <= self.default_quality < self.a_dim):
            raise ValueError(f"default_quality must be in [0, {self.a_dim - 1}]")

        # total chunks derived from trace length (one chunk decision per slot)
        self.total_video_chunks = int(len(self.all_cooked_bw[0]))
        self.chunk_til_end_cap = float(self.total_video_chunks)

        # build approximate chunk sizes (bytes) for each bitrate, each chunk
        # size_bytes = bitrate_kbps * chunk_len_ms / 8
        # (because kbps * ms -> bits)
        self.video_size_by_bitrate = self._build_cbr_video_sizes()

        # network simulator
        self.net_env = Environment(
            all_cooked_time=self.all_cooked_time,
            all_cooked_bw=self.all_cooked_bw,
            all_cooked_dur_ms=self.all_cooked_dur_ms,
            all_cooked_latency_ms=self.all_cooked_latency_ms,
            video_size_by_bitrate=self.video_size_by_bitrate,
            chunk_len_ms=self.chunk_len_ms,
            random_seed=self.random_seed,
            queue_delay_ms=queue_delay_ms,
        )

        # rolling env state
        self.time_stamp = 0.0  # ms
        self.last_bit_rate = self.default_quality
        self.buffer_size = 0.0  # sec
        self.state = np.zeros((S_INFO, S_LEN), dtype=np.float32)

    def _build_cbr_video_sizes(self):
        """
        Creates dict[q] = [chunk_size_bytes] * total_video_chunks
        using constant bitrate approximation.
        """
        out = {}
        for q, br_kbps in enumerate(self.video_bitrates):
            # kbps * ms / 8 = bytes
            chunk_bytes = int(round(float(br_kbps) * self.chunk_len_ms / 8.0))
            out[q] = [chunk_bytes] * self.total_video_chunks
        return out

    def _update_state(
        self,
        bit_rate_idx,
        delay_ms,
        video_chunk_size_bytes,
        next_video_chunk_sizes,
        video_chunk_remain,
    ):
        s = np.roll(self.state, -1, axis=1)
        s[:, -1] = 0.0  # clear newest column

        s[0, -1] = self.video_bitrates[bit_rate_idx] / float(np.max(self.video_bitrates))   # normalized bitrate
        s[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR                                     # buffer / 10s
        s[2, -1] = float(video_chunk_size_bytes) / max(float(delay_ms), 1e-6) / M_IN_K      # KB/ms
        s[3, -1] = float(delay_ms) / M_IN_K / BUFFER_NORM_FACTOR                             # sec/10
        s[4, : self.a_dim] = np.asarray(next_video_chunk_sizes, dtype=np.float32) / M_IN_K / M_IN_K  # MB
        s[5, -1] = min(video_chunk_remain, self.chunk_til_end_cap) / self.chunk_til_end_cap

        self.state = s
        return s

    def seed(self, num):
        self.random_seed = int(num)
        np.random.seed(self.random_seed)
        if hasattr(self.net_env, "seed"):
            self.net_env.seed(self.random_seed)

    def reset(self):
        self.time_stamp = 0.0
        self.last_bit_rate = self.default_quality
        self.buffer_size = 0.0
        self.state = np.zeros((S_INFO, S_LEN), dtype=np.float32)

        if hasattr(self.net_env, "reset_episode"):
            self.net_env.reset_episode()

        bit_rate = self.last_bit_rate
        (
            delay,
            sleep_time,
            self.buffer_size,
            rebuf,
            video_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
        ) = self.net_env.get_video_chunk(bit_rate)

        self.time_stamp += delay + sleep_time

        _ = rebuf, end_of_video  # not used in reset
        return self._update_state(
            bit_rate_idx=bit_rate,
            delay_ms=delay,
            video_chunk_size_bytes=video_chunk_size,
            next_video_chunk_sizes=next_video_chunk_sizes,
            video_chunk_remain=video_chunk_remain,
        )

    def render(self):
        return None

    def step(self, action):
        bit_rate = int(action)
        if not (0 <= bit_rate < self.a_dim):
            raise ValueError(f"action must be in [0, {self.a_dim - 1}]")

        (
            delay,
            sleep_time,
            self.buffer_size,
            rebuf,
            video_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
        ) = self.net_env.get_video_chunk(bit_rate)

        self.time_stamp += delay + sleep_time  # ms

        # reward: quality - rebuffer penalty - smoothness penalty
        reward = (
            self.video_bitrates[bit_rate] / M_IN_K
            - self.rebuf_penalty * rebuf
            - self.smooth_penalty * abs(self.video_bitrates[bit_rate] - self.video_bitrates[self.last_bit_rate]) / M_IN_K
        )

        self.last_bit_rate = bit_rate
        state = self._update_state(
            bit_rate_idx=bit_rate,
            delay_ms=delay,
            video_chunk_size_bytes=video_chunk_size,
            next_video_chunk_sizes=next_video_chunk_sizes,
            video_chunk_remain=video_chunk_remain,
        )

        info = {
            "bitrate_kbps": float(self.video_bitrates[bit_rate]),
            "rebuffer_s": float(rebuf),
            "buffer_s": float(self.buffer_size),
            "delay_ms": float(delay),
            "sleep_ms": float(sleep_time),
            "chunk_size_bytes": int(video_chunk_size),
            "chunks_remain": int(video_chunk_remain),
        }
        return state, float(reward), bool(end_of_video), info