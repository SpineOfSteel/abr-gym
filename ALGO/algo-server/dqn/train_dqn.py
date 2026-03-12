import os, random
import numpy as np
import torch

from SERVER.EnvAbr import ABREnv
import SERVER.dqn.dqn as dqn

# trace json path, video file with bitrate ladder, model path, log path and test script
TRACE_JSON_PATH = "DATASET\\NETWORK\\network.json"  # trace JSON [{duration_ms, bandwidth_kbps, latency_ms}, ...]
VIDEO_PATH = "DATASET\\MOVIE\\movie_4g.json"
SAVE_DIR = "DATASET\\MODELS"
os.makedirs(SAVE_DIR, exist_ok=True)

RANDOM_SEED = 42
MAX_EPISODES = 2000
MAX_STEPS = 1000
LR = 1e-4

EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY_STEPS = 100_000




def main():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(1)

    env = ABREnv(trace_json_path=TRACE_JSON_PATH, video_path=VIDEO_PATH, random_seed=RANDOM_SEED)
    s_dim = [6, 8]
    a_dim = int(env.a_dim)

    agent = dqn.Network(state_dim=s_dim, action_dim=a_dim, learning_rate=LR)

    global_step = 0
    for ep in range(1, MAX_EPISODES + 1):
        obs = env.reset()
        ep_reward = 0.0
        losses = []
        
        for t in range(MAX_STEPS):
            eps = eps_by_step(global_step)

            if np.random.rand() < eps:
                action = np.random.randint(a_dim)
            else:
                q = agent.predict(np.expand_dims(obs, axis=0))  # [A]
                action = int(np.argmax(q))

            next_obs, reward, done, info = env.step(action)

            # IMPORTANT: Network.train(...) expects batches, and p_batch means next-state batch
            loss = agent.train(
                s_batch=[obs],
                a_batch=[one_hot(action, a_dim)],
                p_batch=[next_obs],                    # next-state batch
                r_batch=[[float(reward)]],
                d_batch=[[1.0 if done else 0.0]],
                epoch=ep
            )
            if loss is not None:
                losses.append(loss)

            obs = next_obs
            ep_reward += float(reward)
            global_step += 1

            if done:
                break

        if ep % 10 == 0: print(f"ep={ep}/{1+MAX_EPISODES} steps={t+1} eps={eps:.3f} reward={ep_reward:.3f} "
              f"loss={np.mean(losses) if losses else float('nan'):.4f}")

        if ep % 100 == 0:
            agent.save_model(os.path.join(SAVE_DIR, f"dqn_ep_{ep}.pth"))


    
def eps_by_step(step):
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    return EPS_START + (EPS_END - EPS_START) * (step / EPS_DECAY_STEPS)

def one_hot(a, n):
    x = np.zeros(n, dtype=np.float32)
    x[a] = 1.0
    return x

if __name__ == "__main__":
    main()
