# abr/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PathConfig:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    sabre_entry: Path = field(init=False)
    rl_entry: Path = field(init=False)
    llm_entry: Path = field(init=False)

    network: Path = field(init=False)
    movie: Path = field(init=False)
    chunk_folder: Path = field(init=False)

    train_trace: Path = field(init=False)
    valid_trace: Path = field(init=False)
    test_trace: Path = field(init=False)

    video_meta: Path = field(init=False)
    rl_models_dir: Path = field(init=False)
    tb_log_dir: Path = field(init=False)

    plot_entry: Path = field(init=False)
    plot_source: Path = field(init=False)
    plot_output_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.sabre_entry = self.root / "SIMULATOR" / "SABRE" / "sab.py"
        self.rl_entry = self.root / "algo" / "rl" / "entry.py"
        self.llm_entry = self.root / "algo" / "llm" / "run_plm.py"

        self.network = self.root / "DATASET" / "NETWORK" / "4Glogs_lum" / "logs.parquet"
        self.movie = self.root / "DATASET" / "MOVIE" / "movie_4g.json"
        self.chunk_folder = self.root / "DATASET" / "artifacts" / "tmp"

        self.train_trace = self.root / "DATASET" / "NETWORK" / "train"
        self.valid_trace = self.root / "DATASET" / "NETWORK" / "valid"
        self.test_trace = self.root / "DATASET" / "NETWORK" / "test"

        self.video_meta = self.root / "DATASET" / "MOVIE" / "movie_4g.json"
        self.rl_models_dir = self.root / "DATASET" / "MODELS"
        self.tb_log_dir = self.root / "DATASET" / "tb_logs"

        self.plot_entry = self.root / "PLOT" / "plot_grouped.py"
        self.plot_source = self.root / "DATASET" / "artifacts" / "norway" / "results.all.parquet"
        self.plot_output_dir = self.root / "graphs"
@dataclass
class CommonConfig:
    algorithm: str
    simulator: Optional[str] = None
    paths: PathConfig = field(default_factory=PathConfig)

    network: str = ""
    network_multiplier: float = 1.0
    movie: str = ""
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

    def __post_init__(self) -> None:
        if not self.network:
            self.network = str(self.paths.network)
        if not self.movie:
            self.movie = str(self.paths.movie)
        if not self.chunk_folder:
            self.chunk_folder = str(self.paths.chunk_folder)
            
@dataclass
class RLRewardConfig:
    default_quality: int = 1
    rebuf_penalty: float = 4.3
    smooth_penalty: float = 1.0


@dataclass
class RLEnvConfig:
    flatten_obs: bool = False
    fixed_start_train: bool = False
    fixed_start_eval: bool = True
    action_mode: str = "discrete"
    continuous_map: str = "nearest"
    random_seed: int = 42


@dataclass
class RLRunConfig:
    algorithm: str
    policy: Optional[str] = None
    feature_extractor: Optional[str] = None

    paths: PathConfig = field(default_factory=PathConfig)
    env: RLEnvConfig = field(default_factory=RLEnvConfig)
    reward: RLRewardConfig = field(default_factory=RLRewardConfig)

    train_trace: str = ""
    valid_trace: str = ""
    test_trace: str = ""
    video_meta: str = ""

    model_path: str = ""
    tensorboard_log: str = ""

    total_timesteps: int = 50_000
    n_eval_episodes: int = 10
    deterministic: bool = True
    max_steps: int = 1000
    split: str = "test"

    algo_kwargs: Dict[str, Any] = field(default_factory=dict)
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.train_trace:
            self.train_trace = str(self.paths.train_trace)
        if not self.valid_trace:
            self.valid_trace = str(self.paths.valid_trace)
        if not self.test_trace:
            self.test_trace = str(self.paths.test_trace)
        if not self.video_meta:
            self.video_meta = str(self.paths.video_meta)
        if not self.tensorboard_log:
            self.tensorboard_log = str(self.paths.tb_log_dir)
        if not self.model_path:
            self.model_path = str(self.paths.rl_models_dir / f"{self.algorithm}.zip")