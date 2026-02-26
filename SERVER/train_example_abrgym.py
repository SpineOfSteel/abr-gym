
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from SERVER.EnvAbrGym import AbrGymEnv

# trace json path, video file with bitrate ladder, model path, log path and test script
TRACE_JSON_PATH = "DATASET\\NETWORK\\network.json"  # trace JSON [{duration_ms, bandwidth_kbps, latency_ms}, ...]
VIDEO_PATH = "DATASET\\MOVIE\\movie_4g.json"
SAVE_DIR = "DATASET\\MODELS"
os.makedirs(SAVE_DIR, exist_ok=True)

env = AbrGymEnv(
    trace_json_path=TRACE_JSON_PATH,
    video_path=VIDEO_PATH,
    flatten_obs=True,   # easiest: use MlpPolicy on 48-dim vector
    rebuf_penalty=4.3,
    smooth_penalty=1.0,
    debug=False,
)

check_env(env, warn=True)  # SB3 recommends this :contentReference[oaicite:13]{index=13}

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("ppo_sb3_abr.zip")