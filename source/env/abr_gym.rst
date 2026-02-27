Training with Stable-Baselines3 (PPO example)
=============================================

Below is a minimal end-to-end example showing how to train a policy using
Stable-Baselines3 PPO on :class:`AbrGymEnv`.

Key points from :class:`AbrGymEnv`
---------------------------------

- ``action_space`` is ``Discrete(a_dim)`` (quality index 0..N-1). :contentReference[oaicite:1]{index=1}
- Observations are a rolling Pensieve-style state with shape ``(6,8)`` in ``ABREnv``.
  If ``flatten_obs=True`` (default), the gym wrapper returns a flat vector of shape
  ``(48,)``. :contentReference[oaicite:2]{index=2}
- The wrapper returns Gymnasium’s 5-tuple: ``obs, reward, terminated, truncated, info``. :contentReference[oaicite:3]{index=3}

Example script
--------------

.. code-block:: python

   import os

   from stable_baselines3 import PPO
   from stable_baselines3.common.env_checker import check_env

   from SERVER.EnvAbrGym import AbrGymEnv

   # trace json path, video file with bitrate ladder, model path
   TRACE_JSON_PATH = "DATASET\\NETWORK\\network.json"  # [{duration_ms, bandwidth_kbps, latency_ms}, ...]
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

   # SB3 recommends validating custom envs
   check_env(env, warn=True)

   model = PPO("MlpPolicy", env, verbose=1)
   model.learn(total_timesteps=200_000)

   out_path = os.path.join(SAVE_DIR, "ppo_sb3_abr")
   model.save(out_path)  # produces ppo_sb3_abr.zip

Recommended evaluation pattern
------------------------------

For testing a trained model, create a fresh environment and run a rollout:

.. code-block:: python

   from stable_baselines3 import PPO
   from SERVER.EnvAbrGym import AbrGymEnv

   TRACE_JSON_PATH = "DATASET\\NETWORK\\network.json"
   VIDEO_PATH = "DATASET\\MOVIE\\movie_4g.json"

   env = AbrGymEnv(TRACE_JSON_PATH, VIDEO_PATH, flatten_obs=True)
   model = PPO.load("DATASET\\MODELS\\ppo_sb3_abr.zip", env=env)

   obs, info = env.reset()
   terminated = truncated = False
   total_reward = 0.0

   while not (terminated or truncated):
       action, _ = model.predict(obs, deterministic=True)
       obs, reward, terminated, truncated, info = env.step(action)
       total_reward += reward

   print("Episode total reward:", total_reward)
   print("Last info:", info)

Notes
-----

- If you want a policy that uses the 2D history directly (e.g., Conv1D),
  set ``flatten_obs=False`` and adapt your SB3 policy/network accordingly.
- Your paths use Windows separators (``\\``). On Linux/ReadTheDocs, use
  forward slashes or ``os.path.join``.