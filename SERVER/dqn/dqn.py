import copy
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

FEATURE_NUM = 128
GAMMA = 0.99
TAU = 1e-5
MAX_POOL_NUM = 500_000
ACTION_EPS = 1e-6


class _QNet(nn.Module):
    """
      rows 0,1,5 -> FC(latest scalar)
      rows 2,3   -> Conv1D(k=1) over history
      row 4      -> Conv1D(k=1) over first A_DIM entries
    Input: [B, S_INFO, S_LEN]
    Output: [B, A_DIM] Q-values
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.s_info, self.s_len = int(state_dim[0]), int(state_dim[1])
        self.a_dim = int(action_dim)

        self.fc0 = nn.Linear(1, FEATURE_NUM)
        self.fc1 = nn.Linear(1, FEATURE_NUM)
        self.fc5 = nn.Linear(1, FEATURE_NUM)

        # TF code used conv_1d(..., kernel=1)
        self.conv2 = nn.Conv1d(1, FEATURE_NUM, kernel_size=1)
        self.conv3 = nn.Conv1d(1, FEATURE_NUM, kernel_size=1)
        self.conv4 = nn.Conv1d(1, FEATURE_NUM, kernel_size=1)

        # conv2/3 lengths = S_LEN, conv4 length = A_DIM
        merged_dim = (
            FEATURE_NUM + FEATURE_NUM +
            FEATURE_NUM * self.s_len +
            FEATURE_NUM * self.s_len +
            FEATURE_NUM * self.a_dim +
            FEATURE_NUM
        )
        self.fc_merge = nn.Linear(merged_dim, FEATURE_NUM)
        self.q_head = nn.Linear(FEATURE_NUM, self.a_dim)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected [B,S_INFO,S_LEN], got {tuple(x.shape)}")

        b0 = torch.relu(self.fc0(x[:, 0:1, -1]))          # [B,1] -> [B,128]
        b1 = torch.relu(self.fc1(x[:, 1:2, -1]))
        b5 = torch.relu(self.fc5(x[:, 5:6, -1]))

        b2 = torch.relu(self.conv2(x[:, 2:3, :]))         # [B,128,S_LEN]
        b3 = torch.relu(self.conv3(x[:, 3:4, :]))         # [B,128,S_LEN]
        b4 = torch.relu(self.conv4(x[:, 4:5, :self.a_dim]))  # [B,128,A_DIM]

        merged = torch.cat([
            b0, b1,
            torch.flatten(b2, 1),
            torch.flatten(b3, 1),
            torch.flatten(b4, 1),
            b5,
        ], dim=1)

        z = torch.relu(self.fc_merge(merged))
        return self.q_head(z)  # linear output (Q-values)


class Network:
    """
    Minimal Torch Double-DQN wrapper, intentionally compatible with your older API:
      - predict(state)
      - train(s_batch, a_batch, p_batch, r_batch, d_batch, epoch)   # p_batch = next_state batch
      - get_network_params / set_network_params
      - save_model / load_model
    """
    def __init__(self, state_dim, action_dim, learning_rate,
                 gamma=GAMMA, tau=TAU, replay_size=MAX_POOL_NUM,
                 min_replay=4096, batch_size=1024, device=None):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.min_replay = int(min_replay)
        self.batch_size = int(batch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.eval_net = _QNet(state_dim, action_dim).to(self.device)
        self.target_net = _QNet(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.mse = nn.MSELoss()

        self.pool = deque(maxlen=int(replay_size))

    # ---------- inference ----------
    def predict(self, x):
        self.eval_net.eval()
        x_t = _to_tensor(x, self.device)
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(0)
        with torch.no_grad():
            q = self.eval_net(x_t)
        q = q.detach().cpu().numpy()
        return q[0] if q.shape[0] == 1 else q

    # ---------- replay + training ----------
    def _push_batch_to_replay(self, s_batch, a_batch, ns_batch, r_batch, d_batch):
        for s, a, ns, r, d in zip(s_batch, a_batch, ns_batch, r_batch, d_batch):
            self.pool.append((
                np.asarray(s, dtype=np.float32),
                np.asarray(a, dtype=np.float32),
                np.asarray(ns, dtype=np.float32),
                float(np.asarray(r).reshape(-1)[0]),
                float(np.asarray(d).reshape(-1)[0]),
            ))

    def _sample_replay(self):
        idx = np.random.randint(0, len(self.pool), size=self.batch_size)
        batch = [self.pool[i] for i in idx]

        s = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
        a = np.stack([b[1] for b in batch], axis=0).astype(np.float32)   # one-hot
        ns = np.stack([b[2] for b in batch], axis=0).astype(np.float32)
        r = np.array([b[3] for b in batch], dtype=np.float32).reshape(-1, 1)
        d = np.array([b[4] for b in batch], dtype=np.float32).reshape(-1, 1)
        return s, a, ns, r, d

    def update_target(self, tau=None):
        t = self.tau if tau is None else float(tau)
        with torch.no_grad():
            for tp, ep in zip(self.target_net.parameters(), self.eval_net.parameters()):
                tp.mul_(1.0 - t).add_(ep, alpha=t)

    def train(self, s_batch, a_batch, p_batch, r_batch, d_batch, epoch=None):
        """
        Compatibility note:
          p_batch = next_state batch (same naming confusion as old TF code)
        """
        # Store transitions
        self._push_batch_to_replay(s_batch, a_batch, p_batch, r_batch, d_batch)

        if len(self.pool) < self.min_replay:
            return None  # warmup period

        s, a, ns, r, d = self._sample_replay()

        s_t = _to_tensor(s, self.device)
        a_t = _to_tensor(a, self.device)
        ns_t = _to_tensor(ns, self.device)
        r_t = _to_tensor(r, self.device)
        d_t = _to_tensor(d, self.device)

        self.eval_net.train()
        q_eval = self.eval_net(s_t)                         # [B, A]
        q_eval_selected = torch.sum(q_eval * a_t, dim=1, keepdim=True)

        # Double DQN target:
        # next action from eval_net, value from target_net
        with torch.no_grad():
            q_eval_ns = self.eval_net(ns_t)                 # [B, A]
            next_action = torch.argmax(q_eval_ns, dim=1, keepdim=True)  # [B,1]
            q_target_ns = self.target_net(ns_t)             # [B, A]
            q_next = torch.gather(q_target_ns, 1, next_action)          # [B,1]
            y = r_t + self.gamma * (1.0 - d_t) * q_next

        loss = self.mse(q_eval_selected, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target()

        return float(loss.detach().cpu().item())

    # ---------- params / checkpoints ----------
    def get_network_params(self):
        return [_cpu_state_dict(self.eval_net.state_dict()),
                _cpu_state_dict(self.target_net.state_dict())]

    def set_network_params(self, params):
        eval_sd, target_sd = _parse_ckpt(params)
        self.eval_net.load_state_dict(eval_sd)
        if target_sd is None:
            self.target_net.load_state_dict(eval_sd)
        else:
            self.target_net.load_state_dict(target_sd)

    def save_model(self, path):
        torch.save(self.get_network_params(), path)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.set_network_params(ckpt)
        


def _to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _cpu_state_dict(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def _parse_ckpt(ckpt):
    # Supports [eval_sd, target_sd] or dict form
    if isinstance(ckpt, (list, tuple)) and len(ckpt) >= 2:
        return ckpt[0], ckpt[1]
    if isinstance(ckpt, dict):
        if "eval" in ckpt and "target" in ckpt:
            return ckpt["eval"], ckpt["target"]
        if "eval_state_dict" in ckpt and "target_state_dict" in ckpt:
            return ckpt["eval_state_dict"], ckpt["target_state_dict"]
        # fallback: eval-only
        return ckpt, None
    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

