# abr/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class PathConfig:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    @property
    def sabre_entry(self) -> Path:
        return self.root / "SIMULATOR" / "SABRE" / "sab.py"

    @property
    def llm_entry(self) -> Path:
        return self.root / "ALGO" / "llm" / "run_plm.py"

    @property
    def default_network(self) -> Path:
        return self.root / "DATASET" / "NETWORK" / "4Glogs_lum" / "logs.parquet"

    @property
    def default_movie(self) -> Path:
        return self.root / "DATASET" / "MOVIE" / "movie_4g.json"

    @property
    def default_chunk_folder(self) -> Path:
        return self.root / "DATASET" / "artifacts" / "tmp"


@dataclass
class CommonConfig:
    algorithm: str
    simulator: Optional[str] = None

    network: Optional[str] = None
    network_multiplier: float = 1.0
    movie: Optional[str] = None
    movie_length: Optional[float] = None

    plugin: List[str] = field(default_factory=list)
    moving_average: str = "ewma"
    window_size: List[int] = field(default_factory=lambda: [3])
    half_life: List[float] = field(default_factory=lambda: [3.0, 8.0])

    seek: Optional[List[float]] = None
    replace: str = "none"
    max_buffer: float = 25.0
    no_abandon: bool = False
    rampup_threshold: Optional[int] = None
    gamma_p: float = 5.0
    no_insufficient_buffer_rule: bool = False

    verbose: bool = True
    chunk_log: str = "log.txt"
    chunk_folder: str = ""
    chunk_log_start_ts: Optional[float] = 1608418125.0

    shim: int = 8333
    timeout_s: float = 1.0
    debug_p: bool = False
    ping_on_start: bool = False

    seed: int = 100003
    device: Optional[str] = None

    paths: PathConfig = field(default_factory=PathConfig)

    def finalize_defaults(self) -> None:
        if self.network is None:
            self.network = str(self.paths.default_network)
        if self.movie is None:
            self.movie = str(self.paths.default_movie)