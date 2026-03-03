import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---- constants ----
GAMMA = 0.99
A_DIM = 6
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6
ACTION_EPS = 1e-6

class _A3CBody(nn.Module):
    """
    feature extractor:
      rows 0,1,5 -> FC(latest scalar)
      rows 2,3   -> Conv1D(history length S_LEN)
      row 4      -> Conv1D(first A_DIM entries)
    Input: [B, S_INFO, S_LEN]
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.s_info, self.s_len = int(state_dim[0]), int(state_dim[1])
        self.a_dim = int(action_dim)

        if self.s_info < 6: raise ValueError(f"S_INFO must be >= 6, got {self.s_info}")
        if self.s_len < 4: raise ValueError(f"S_LEN must be >= 4, got {self.s_len}")
        if self.a_dim < 4: raise ValueError(f"A_DIM must be >= 4, got {self.a_dim}")

        self.fc0 = nn.Linear(1, 128)
        self.fc1 = nn.Linear(1, 128)
        self.fc5 = nn.Linear(1, 128)

        self.conv2 = nn.Conv1d(1, 128, kernel_size=4)
        self.conv3 = nn.Conv1d(1, 128, kernel_size=4)
        self.conv4 = nn.Conv1d(1, 128, kernel_size=4)

        conv23_len = self.s_len - 4 + 1
        conv4_len = self.a_dim - 4 + 1
        merged_dim = 128 + 128 + 128 * conv23_len + 128 * conv23_len + 128 * conv4_len + 128
        self.fc_merge = nn.Linear(merged_dim, 128)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected [B,S_INFO,S_LEN], got {tuple(x.shape)}")

        b0 = torch.relu(self.fc0(x[:, 0:1, -1]))
        b1 = torch.relu(self.fc1(x[:, 1:2, -1]))
        b5 = torch.relu(self.fc5(x[:, 5:6, -1]))

        b2 = torch.relu(self.conv2(x[:, 2:3, :]))
        b3 = torch.relu(self.conv3(x[:, 3:4, :]))
        b4 = torch.relu(self.conv4(x[:, 4:5, :self.a_dim]))

        merged = torch.cat([
            b0, b1,
            torch.flatten(b2, 1),
            torch.flatten(b3, 1),
            torch.flatten(b4, 1),
            b5
        ], dim=1)

        return torch.relu(self.fc_merge(merged))


class _Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.body = _A3CBody(state_dim, action_dim)
        self.actor_head = nn.Linear(128, action_dim) #head vs actor_head

    def forward(self, x):
        probs = torch.softmax(self.actor_head(self.body(x)), dim=1)
        probs = torch.clamp(probs, ACTION_EPS, 1.0 - ACTION_EPS)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


class _Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.body = _A3CBody(state_dim, action_dim)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        return self.critic_head(self.body(x))


class ActorNetwork:
    def __init__(self, state_dim, action_dim, learning_rate, device=None, verbose=False):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.verbose = verbose
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = _Actor(state_dim, action_dim).to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr_rate)

    def _loss(self, inputs, acts, adv):
        x = _to_tensor(inputs, self.device)
        acts = _to_tensor(acts, self.device)
        adv = _to_tensor(adv, self.device)

        if x.dim() == 2:
            x = x.unsqueeze(0)
        if acts.dim() == 1:
            acts = acts.view(1, -1)
        if adv.dim() == 1:
            adv = adv.view(-1, 1)

        probs = self.model(x)
        selected = torch.sum(probs * acts, dim=1, keepdim=True).clamp_min(ENTROPY_EPS)

        policy_term = torch.sum(torch.log(selected) * (-adv))
        entropy_term = ENTROPY_WEIGHT * torch.sum(probs * torch.log(probs.clamp_min(ENTROPY_EPS)))
        return policy_term + entropy_term

    def train(self, inputs, acts, act_grad_weights):
        self.model.train()
        loss = self._loss(inputs, acts, act_grad_weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def predict(self, inputs):
        self.model.eval()
        x = _to_tensor(inputs, self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
        return out.detach().cpu().numpy()

    def get_gradients(self, inputs, acts, act_grad_weights):
        self.model.train()
        loss = self._loss(inputs, acts, act_grad_weights)
        self.optimizer.zero_grad()
        loss.backward()
        return [None if p.grad is None else p.grad.detach().cpu().numpy().copy()
                for p in self.model.parameters()]

    def apply_gradients(self, actor_gradients):
        self.optimizer.zero_grad()
        for p, g in zip(self.model.parameters(), actor_gradients):
            if g is None:
                continue
            p.grad = torch.as_tensor(g, dtype=torch.float32, device=self.device)
        self.optimizer.step()

    def get_network_params(self):
        return [p.detach().cpu().numpy().copy() for p in self.model.parameters()]

    def set_network_params(self, params):
        with torch.no_grad():
            for p, src in zip(self.model.parameters(), params):
                p.copy_(torch.as_tensor(src, dtype=torch.float32, device=self.device))

    def save(self, path):
        torch.save(_cpu_state_dict(self.model.state_dict()), path)

    def load(self, path, map_location=None):
        ckpt = _torch_load(path, map_location or self.device)
        actor_sd, _ = _parse_checkpoint(ckpt)
        self.model.load_state_dict(actor_sd)


class CriticNetwork:
    def __init__(self, state_dim, learning_rate, action_dim=A_DIM, device=None, verbose=False):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.verbose = verbose
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = _Critic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr_rate)
        self.mse = nn.MSELoss()

    def _loss(self, inputs, td_target):
        x = _to_tensor(inputs, self.device)
        y = _to_tensor(td_target, self.device)

        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.view(-1, 1)

        return self.mse(self.model(x), y)

    def train(self, inputs, td_target):
        self.model.train()
        loss = self._loss(inputs, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item()), None

    def predict(self, inputs):
        self.model.eval()
        x = _to_tensor(inputs, self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
        return out.detach().cpu().numpy()

    def get_td(self, inputs, td_target):
        self.model.eval()
        x = _to_tensor(inputs, self.device)
        y = _to_tensor(td_target, self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.view(-1, 1)
        with torch.no_grad():
            td = y - self.model(x)
        return td.detach().cpu().numpy()

    def get_gradients(self, inputs, td_target):
        self.model.train()
        loss = self._loss(inputs, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        return [None if p.grad is None else p.grad.detach().cpu().numpy().copy()
                for p in self.model.parameters()]

    def apply_gradients(self, critic_gradients):
        self.optimizer.zero_grad()
        for p, g in zip(self.model.parameters(), critic_gradients):
            if g is None:
                continue
            p.grad = torch.as_tensor(g, dtype=torch.float32, device=self.device)
        self.optimizer.step()

    def get_network_params(self):
        return [p.detach().cpu().numpy().copy() for p in self.model.parameters()]

    def set_network_params(self, params):
        with torch.no_grad():
            for p, src in zip(self.model.parameters(), params):
                p.copy_(torch.as_tensor(src, dtype=torch.float32, device=self.device))

    def save(self, path):
        torch.save(_cpu_state_dict(self.model.state_dict()), path)

    def load(self, path, map_location=None):
        ckpt = _torch_load(path, map_location or self.device)
        _, critic_sd = _parse_checkpoint(ckpt)
        if critic_sd is None:
            raise ValueError("Checkpoint does not contain critic weights")
        self.model.load_state_dict(critic_sd)


class Network:
    """
      - predict()
      - train(s,a,p,v,epoch)
      - compute_v(...)
      - save_model/load_model()
      - get/set_network_params()
    """
    def __init__(self, state_dim, action_dim, learning_rate, critic_learning_rate=1e-3, device=None, verbose=False):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self._entropy_weight = ENTROPY_WEIGHT  # compatibility field

        self.actor = ActorNetwork(state_dim, action_dim, learning_rate, device=device, verbose=verbose)
        self.critic = CriticNetwork(state_dim, critic_learning_rate, action_dim=action_dim, device=device, verbose=verbose)

    def predict(self, inputs):
        probs = self.actor.predict(inputs)
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim == 2 and probs.shape[0] == 1:
            probs = probs[0]
        probs = np.clip(probs, ACTION_EPS, 1.0)
        s = float(np.sum(probs))
        if not np.isfinite(s) or s <= 0:
            probs = np.ones(self.a_dim, dtype=np.float32) / float(self.a_dim)
        else:
            probs = probs / s
        return probs

    def train(self, s_batch, a_batch, p_batch, v_batch, epoch=None):
        # p_batch is ignored (A3C update), kept for API compatibility
        s_batch = np.asarray(s_batch, dtype=np.float32)
        a_batch = np.asarray(a_batch, dtype=np.float32)
        R_batch = np.asarray(v_batch, dtype=np.float32).reshape(-1, 1)

        values = self.critic.predict(s_batch)
        adv = R_batch - values

        actor_loss = self.actor.train(s_batch, a_batch, adv)
        critic_loss, _ = self.critic.train(s_batch, R_batch)
        return actor_loss, critic_loss

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        s_batch = np.asarray(s_batch, dtype=np.float32)
        r_batch = np.asarray(r_batch, dtype=np.float32).reshape(-1, 1)

        if len(r_batch) == 0:
            return []

        R_batch = np.zeros_like(r_batch, dtype=np.float32)
        if terminal:
            R_batch[-1, 0] = float(r_batch[-1, 0])
        else:
            v = self.critic.predict(s_batch)
            R_batch[-1, 0] = float(v[-1, 0])

        for t in reversed(range(len(r_batch) - 1)):
            R_batch[t, 0] = float(r_batch[t, 0]) + GAMMA * float(R_batch[t + 1, 0])

        return list(R_batch)

    def get_network_params(self):
        return [_cpu_state_dict(self.actor.model.state_dict()),
                _cpu_state_dict(self.critic.model.state_dict())]

    def set_network_params(self, params):
        actor_sd, critic_sd = _parse_checkpoint(params)
        self.actor.model.load_state_dict(actor_sd)
        if critic_sd is not None:
            self.critic.model.load_state_dict(critic_sd)

    def load_model(self, path):
        ckpt = _torch_load(path, map_location=self.actor.device)
        actor_sd, critic_sd = _parse_checkpoint(ckpt)
        self.actor.model.load_state_dict(actor_sd)
        if critic_sd is not None:
            self.critic.model.load_state_dict(critic_sd)

    def save_model(self, path):
        torch.save(self.get_network_params(), path)


def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):
    s_batch = np.asarray(s_batch, dtype=np.float32)
    a_batch = np.asarray(a_batch, dtype=np.float32)
    r_batch = np.asarray(r_batch, dtype=np.float32).reshape(-1, 1)

    assert s_batch.shape[0] == a_batch.shape[0] == r_batch.shape[0]
    N = s_batch.shape[0]

    v_batch = critic.predict(s_batch)
    R_batch = np.zeros_like(r_batch, dtype=np.float32)

    if terminal:
        R_batch[-1, 0] = 0.0
    else:
        R_batch[-1, 0] = float(v_batch[-1, 0])

    for t in reversed(range(N - 1)):
        R_batch[t, 0] = float(r_batch[t, 0]) + GAMMA * float(R_batch[t + 1, 0])

    td_batch = R_batch - v_batch
    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
    critic_gradients = critic.get_gradients(s_batch, R_batch)
    return actor_gradients, critic_gradients, td_batch


def discount(x, gamma):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    out = np.zeros_like(x)
    if len(x) == 0:
        return out
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    return out


def compute_entropy(x):
    H = 0.0
    for p in np.asarray(x, dtype=np.float32).reshape(-1):
        if 0.0 < p < 1.0:
            H -= float(p) * float(np.log(p))
    return float(H)


def build_summaries():
    return None, ["TD_loss", "Eps_total_reward", "Avg_entropy"]


#HELPERS

def _to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=dtype, device=device)


def _torch_load(path, map_location):
    # torch versions differ on weights_only kwarg
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _cpu_state_dict(sd):
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)
    return out


def _parse_checkpoint(ckpt):
    """
    Supports:
      - [actor_sd, critic_sd]
      - {"actor":..., "critic":...}
      - {"actor_state_dict":..., "critic_state_dict":...}
      - actor-only state_dict
    Returns (actor_sd, critic_sd_or_none)
    """
    if isinstance(ckpt, (list, tuple)) and len(ckpt) >= 2:
        return ckpt[0], ckpt[1]
    if isinstance(ckpt, dict):
        if "actor" in ckpt:
            return ckpt["actor"], ckpt.get("critic")
        if "actor_state_dict" in ckpt:
            return ckpt["actor_state_dict"], ckpt.get("critic_state_dict")
        # plain actor-only state dict
        return ckpt, None
    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")
