# @title AbrGymEnv
# 
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from SERVER.envs.rl.Env import EnvConfig, Environment
from SERVER.envs.rl.loader import load_trace, load_video_size

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1_000_000.0
BITS_IN_BYTE = 8.0

S_INFO = 6
S_LEN = 8
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0


class AbrGymEnv(gym.Env):
    def __init__(
        self,
        trace_path: str,
        video_metadata_file: str,
        random_seed: int = 42,
        default_quality: int = 1,
        rebuf_penalty: float = 4.3,
        smooth_penalty: float = 1.0,
        flatten_obs: bool = True,
        continuous_actions: bool = False,
        env_config: EnvConfig | None = None,
    ):
        super().__init__()

        self.flatten_obs = bool(flatten_obs)
        self.continuous_actions = bool(continuous_actions)
        self.random_seed = int(random_seed)
        self.default_quality = int(default_quality)
        self.rebuf_penalty = float(rebuf_penalty)
        self.smooth_penalty = float(smooth_penalty)

        # Load traces
        all_cooked_time, all_cooked_bw, all_file_names = load_trace(trace_path)
        print(f'Loaded {len(all_file_names)} traces')
        print(f'First trace file: {all_file_names[0] if all_file_names else "N/A"}')
        print(f'Last trace file: {all_file_names[-1] if all_file_names else "N/A"}')

        # Load video metadata
        self.video_size, meta = load_video_size(video_metadata_file, return_unit="bytes")
        self.video_bitrates = np.asarray(meta["bitrates_kbps"], dtype=np.float32)
        self.a_dim = int(meta["bitrate_levels"])
        self.total_video_chunks = int(meta["total_video_chunk"])
        self.chunk_len_ms = float(meta["segment_duration_ms"])
        self.chunk_til_end_cap = float(self.total_video_chunks)
        print(meta)
        if not (0 <= self.default_quality < self.a_dim):
            raise ValueError(f"default_quality must be in [0, {self.a_dim - 1}]")

        # Simple config setup
        self.env_config = env_config or EnvConfig()
        self.env_config.random_seed = self.random_seed
        self.env_config.video_chunk_len_ms = self.chunk_len_ms
        self.env_config.bitrate_levels = self.a_dim
        self.env_config.total_video_chunk = self.total_video_chunks
        self.env_config.video_size_unit = "bytes"
        self.env_config.fixed_start = False #training
        #mark print(self.env_config.cfg())

        # Network environment
        self.net_env = Environment(
            all_cooked_time=all_cooked_time,
            all_cooked_bw=all_cooked_bw, 
            video_size_by_bitrate=self.video_size,
            config=self.env_config,           
        )

        # RL state
        self.time_stamp = 0.0
        self.last_bit_rate = self.default_quality
        self.buffer_size = 0.0
        self.state = np.zeros((S_INFO, S_LEN), dtype=np.float32)

        # Action space
        if self.continuous_actions:
            self.action_space = spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([float(self.a_dim - 1)], dtype=np.float32),
                shape=(1,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Discrete(self.a_dim)

        # Observation space
        obs_shape = (S_INFO * S_LEN,) if self.flatten_obs else (S_INFO, S_LEN)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )

    def seed(self, seed=None):
        if seed is None:
            seed = self.random_seed
        self.random_seed = int(seed)
        np.random.seed(self.random_seed)
        self.env_config.random_seed = self.random_seed

    def reset(self, *, seed=None, options=None):
        del options

        if seed is not None:
            self.seed(seed)

        # Recreate net env for a clean episode
        self.net_env = Environment(
            all_cooked_time=self.net_env.all_cooked_time,
            all_cooked_bw=self.net_env.all_cooked_bw,
            config=self.env_config,
            video_size_by_bitrate=self.video_size,
        )

        self.time_stamp = 0.0
        self.last_bit_rate = self.default_quality
        self.buffer_size = 0.0
        self.state = np.zeros((S_INFO, S_LEN), dtype=np.float32)

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

        state = self._update_state(
            bit_rate,
            delay,
            video_chunk_size,
            next_video_chunk_sizes,
            video_chunk_remain,
        )

        info = {
            "bitrate_kbps": float(self.video_bitrates[bit_rate]),
            "rebuffer_s": float(rebuf),
            "buffer_s": float(self.buffer_size),
            "delay_ms": float(delay),
            "sleep_ms": float(sleep_time),
            "chunk_size_bytes": int(video_chunk_size),
            "chunks_remain": int(video_chunk_remain),
            "end_of_video": bool(end_of_video),
        }

        return self._format_obs(state), info


    def step(self, action):
        bit_rate = self._map_action(action)

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

        reward = (
            self.video_bitrates[bit_rate] / M_IN_K
            - self.rebuf_penalty * rebuf
            - self.smooth_penalty
            * abs(self.video_bitrates[bit_rate] - self.video_bitrates[self.last_bit_rate]) / M_IN_K
        )

        self.last_bit_rate = bit_rate

        state = self._update_state(
            bit_rate,
            delay,
            video_chunk_size,
            next_video_chunk_sizes,
            video_chunk_remain,
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

        terminated = bool(end_of_video)
        truncated = False
        return self._format_obs(state), float(reward), terminated, truncated, info

    def render(self):
        return None

    def _format_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return obs.reshape(-1) if self.flatten_obs else obs

    def _update_state(self, bit_rate, delay, video_chunk_size, next_video_chunk_sizes, video_chunk_remain):
        state = np.roll(self.state, -1, axis=1)

        state[0, -1] = self.video_bitrates[bit_rate] / float(np.max(self.video_bitrates))
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = float(video_chunk_size) / max(float(delay), 1e-6) / M_IN_K
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        state[4, :] = 0.0
        state[4, : self.a_dim] = np.asarray(next_video_chunk_sizes, dtype=np.float32) / M_IN_K / M_IN_K
        state[5, -1] = min(video_chunk_remain, self.chunk_til_end_cap) / self.chunk_til_end_cap

        self.state = state.astype(np.float32)
        return self.state

    def _map_action(self, action):
        if self.continuous_actions:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            return int(np.clip(np.round(action[0]), 0, self.a_dim - 1))
        return int(action)
