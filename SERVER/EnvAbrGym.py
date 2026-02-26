import numpy as np
import gymnasium as gym
from gymnasium import spaces

from SERVER.EnvAbr import ABREnv


class AbrGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, trace_json_path: str, video_path: str, flatten_obs: bool = True, **abr_kwargs):
        super().__init__()
        self.flatten_obs = bool(flatten_obs)

        self.env = ABREnv(
            trace_json_path=trace_json_path,
            video_path=video_path,
            **abr_kwargs,
        )

        # Actions: quality index 0..a_dim-1
        self.action_space = spaces.Discrete(int(self.env.a_dim))

        # Observations: ABREnv maintains a rolling state [6,8] float32
        obs_shape = (6 * 8,) if self.flatten_obs else (6, 8)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )

    def _format_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        return obs.reshape(-1) if self.flatten_obs else obs

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()
        return self._format_obs(obs), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(int(action))
        terminated = bool(done)
        truncated = False
        return self._format_obs(obs), float(reward), terminated, truncated, info