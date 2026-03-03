import numpy as np
from typing import Dict



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


