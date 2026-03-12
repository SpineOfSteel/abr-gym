# @title EnvConfig & Environment
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1_000_000.0
BITS_IN_BYTE = 8.0


@dataclass
class EnvConfig:
    video_chunk_len_ms: float = 1000.0
    bitrate_levels: int = 1
    total_video_chunk: int = 1
    buffer_thresh_ms: float = 60.0 * MILLISECONDS_IN_SECOND
    drain_buffer_sleep_time_ms: float = 500.0
    packet_payload_portion: float = 0.95
    link_rtt_ms: float = 80.0
    noise_low: float = 0.9
    noise_high: float = 1.1
    random_seed: int = 42
    fixed_start: bool = False
    add_noise: bool = True
    video_size_unit: str = 'bytes'  # 'bytes' or 'bits'



class Environment:
    def __init__(
        self,
        all_cooked_time: List[List[float]],
        all_cooked_bw: List[List[float]],
        video_size_by_bitrate: Dict[int, List[int]],
        config: Optional[EnvConfig] = None,        
    ) -> None:
        if len(all_cooked_time) != len(all_cooked_bw) or not all_cooked_time:
            raise ValueError('all_cooked_time and all_cooked_bw must be non-empty and aligned')

        self.config = config or EnvConfig()
        self.rng = np.random.RandomState(self.config.random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw
        
        self.video_size = video_size_by_bitrate
        self.video_size_unit = self.config.video_size_unit
        self.bitrate_levels = len(video_size_by_bitrate)
        self.total_video_chunk = len(next(iter(video_size_by_bitrate.values())))
        self.video_chunk_len_ms = int(self.config.video_chunk_len_ms)

        self.config.bitrate_levels = self.bitrate_levels
        self.config.total_video_chunk = self.total_video_chunk
        self.config.video_chunk_len_ms = self.video_chunk_len_ms

        self.video_chunk_counter = 0
        self.buffer_size = 0.0

        self.trace_idx = 0 if self.config.fixed_start else self._r(0,len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.mahimahi_start_ptr = 1
        self.mahimahi_ptr = 1 if self.config.fixed_start else self._r(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

    def _r(self, st,en):
        return  int(self.rng.randint(st, en))

    def cfg(self):
        return self.config

    def _advance_trace_for_next_video(self) -> None:
        if self.config.fixed_start:
            self.trace_idx = (self.trace_idx + 1) % len(self.all_cooked_time)
        else:
            self.trace_idx = self._r(0,len(self.all_cooked_time))

        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        
        self.mahimahi_ptr = self.mahimahi_start_ptr if self.config.fixed_start else self._r(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

    def get_video_chunk(self, quality: int):
        assert 0 <= quality < self.bitrate_levels

        stored_chunk_size = self.video_size[quality][self.video_chunk_counter]
        chunk_size_bytes = (
            float(stored_chunk_size)
            if self.video_size_unit == 'bytes'
            else float(stored_chunk_size) / BITS_IN_BYTE
        )

        delay = 0.0
        video_chunk_counter_sent = 0.0

        while True:
            throughput = self.cooked_bw[self.mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time
            packet_payload = throughput * duration * self.config.packet_payload_portion

            if video_chunk_counter_sent + packet_payload > chunk_size_bytes:
                fractional_time = (
                    (chunk_size_bytes - video_chunk_counter_sent)
                    / throughput
                    / self.config.packet_payload_portion
                )
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += self.config.link_rtt_ms

        if self.config.add_noise:
            delay *= float(self.rng.uniform(self.config.noise_low, self.config.noise_high))

        rebuf = max(delay - self.buffer_size, 0.0)
        self.buffer_size = max(self.buffer_size - delay, 0.0)
        self.buffer_size += self.video_chunk_len_ms

        sleep_time = 0.0
        if self.buffer_size > self.config.buffer_thresh_ms:
            drain_buffer_time = self.buffer_size - self.config.buffer_thresh_ms
            sleep_time = (
                np.ceil(drain_buffer_time / self.config.drain_buffer_sleep_time_ms)
                * self.config.drain_buffer_sleep_time_ms
            )
            self.buffer_size -= sleep_time

            remaining_sleep = sleep_time
            while True:
                duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time
                if duration > remaining_sleep / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += remaining_sleep / MILLISECONDS_IN_SECOND
                    break

                remaining_sleep -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = self.total_video_chunk - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.total_video_chunk:
            end_of_video = True
            self.buffer_size = 0.0
            self.video_chunk_counter = 0
            self._advance_trace_for_next_video()

        next_video_chunk_sizes = [
            self.video_size[i][self.video_chunk_counter] for i in range(self.bitrate_levels)
        ]

        return (
            delay,
            sleep_time,
            return_buffer_size / MILLISECONDS_IN_SECOND,
            rebuf / MILLISECONDS_IN_SECOND,
            stored_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
        )


