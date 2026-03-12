
from pathlib import Path
from typing import Optional
import os


M_IN_K = 1000.0
REBUF_PENALTY = 32.0  #160.0
SMOOTH_PENALTY = 1.0

def compute_qoe_reward(bitrate_kbps: float, stall_s: float, last_bitrate_kbps: float | None) -> float:
    bitrate_mbps = bitrate_kbps / M_IN_K
    last_mbps = (last_bitrate_kbps / M_IN_K) if last_bitrate_kbps is not None else 0.0
    return bitrate_mbps - REBUF_PENALTY * stall_s - SMOOTH_PENALTY * abs(bitrate_mbps - last_mbps)


class ChunkLogger:
    def __init__(self, chunk_folder: str, chunk_log:str, start_ts: Optional[float]):
        print(os.getcwd(), chunk_folder, chunk_log)
        dir = Path.cwd() / chunk_folder
        dir.mkdir(parents=True, exist_ok=True)
        print('OUT>>', dir)
        
        self.f = open(f'{dir}//{chunk_log}', "w", encoding="utf-8")
        self.start_ts = start_ts
        self.last_bitrate_kbps: Optional[float] = None

    def close(self) -> None:
        self.f.close()

    def log(self, sim_time_ms: float, bitrate_kbps: float, buffer_ms: float, stall_ms: float,
            chunk_bits: float, download_ms: float) -> None:
        ts_s = (self.start_ts + sim_time_ms / 1000.0) if self.start_ts is not None else (sim_time_ms / 1000.0)
        stall_s = stall_ms / 1000.0
        reward = compute_qoe_reward(bitrate_kbps, stall_s, self.last_bitrate_kbps)
        self.last_bitrate_kbps = bitrate_kbps

        chunk_bytes = chunk_bits / 8.0
        buf_s = buffer_ms / 1000.0
        
        
        self.f.write(
            f"{ts_s:.2f}\t{bitrate_kbps:.0f}\t{buf_s:.6f}\t{stall_s:.3f}\t"
            f"{chunk_bytes:.0f}\t{download_ms:.0f}\t{reward:.12g}\n"
        )
