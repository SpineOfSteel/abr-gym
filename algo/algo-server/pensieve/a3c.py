import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from SERVER.pensieve.ac import Actor, Critic

# ---- constants ----
GAMMA = 0.99
A_DIM = 6
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6
ACTION_EPS = 1e-6

class Network:
    def __init__(self, state_dim, action_dim, learning_rate, critic_learning_rate=1e-3, device=None, verbose=False):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self._entropy_weight = ENTROPY_WEIGHT
        self.actor = ActorNetwork(state_dim, action_dim, learning_rate, device=device, verbose=verbose)
        self.critic = CriticNetwork(state_dim, critic_learning_rate, action_dim=action_dim, device=device, verbose=verbose)

    def predict(self, inputs):
        probs = np.asarray(self.actor.predict(inputs), dtype=np.float32)
        if probs.ndim == 2 and probs.shape[0] == 1:
            probs = probs[0]
        probs = np.clip(probs, ACTION_EPS, 1.0)
        probs_sum = float(np.sum(probs))
        if not np.isfinite(probs_sum) or probs_sum <= 0:
            return np.ones(self.a_dim, dtype=np.float32) / float(self.a_dim)
        return probs / probs_sum

    def train(self, s_batch, a_batch, p_batch, v_batch, epoch=None):
        s_batch = np.asarray(s_batch, dtype=np.float32)
        a_batch = np.asarray(a_batch, dtype=np.float32)
        R_batch = np.asarray(v_batch, dtype=np.float32).reshape(-1, 1)

        values = self.critic.predict(s_batch)
        advantages = R_batch - values

        actor_loss = self.actor.train(s_batch, a_batch, advantages)
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
            value = self.critic.predict(s_batch)
            R_batch[-1, 0] = float(value[-1, 0])

        for t in reversed(range(len(r_batch) - 1)):
            R_batch[t, 0] = float(r_batch[t, 0]) + GAMMA * float(R_batch[t + 1, 0])
        return list(R_batch)

    def get_network_params(self):
        return [_cpu_state_dict(self.actor.model.state_dict()), _cpu_state_dict(self.critic.model.state_dict())]

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
        

class ActorNetwork:
    def __init__(self, state_dim, action_dim, learning_rate, device=None, verbose=False):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.verbose = verbose
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = Actor(state_dim, action_dim).to(self.device)
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
        policy_loss = -torch.sum(torch.log(selected) * adv)
        entropy_loss = ENTROPY_WEIGHT * torch.sum(probs * torch.log(probs.clamp_min(ENTROPY_EPS)))
        return policy_loss + entropy_loss

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
            return self.model(x).detach().cpu().numpy()

    def get_gradients(self, inputs, acts, act_grad_weights):
        self.model.train()
        loss = self._loss(inputs, acts, act_grad_weights)
        self.optimizer.zero_grad()
        loss.backward()
        return [None if p.grad is None else p.grad.detach().cpu().numpy().copy() for p in self.model.parameters()]

    def apply_gradients(self, actor_gradients):
        self.optimizer.zero_grad()
        for p, g in zip(self.model.parameters(), actor_gradients):
            if g is not None:
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
        self.model = Critic(state_dim, action_dim).to(self.device)
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
            return self.model(x).detach().cpu().numpy()

    def get_td(self, inputs, td_target):
        self.model.eval()
        x = _to_tensor(inputs, self.device)
        y = _to_tensor(td_target, self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.view(-1, 1)
        with torch.no_grad():
            return (y - self.model(x)).detach().cpu().numpy()

    def get_gradients(self, inputs, td_target):
        self.model.train()
        loss = self._loss(inputs, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        return [None if p.grad is None else p.grad.detach().cpu().numpy().copy() for p in self.model.parameters()]

    def apply_gradients(self, critic_gradients):
        self.optimizer.zero_grad()
        for p, g in zip(self.model.parameters(), critic_gradients):
            if g is not None:
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




def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):
    s_batch = np.asarray(s_batch, dtype=np.float32)
    a_batch = np.asarray(a_batch, dtype=np.float32)
    r_batch = np.asarray(r_batch, dtype=np.float32).reshape(-1, 1)

    assert s_batch.shape[0] == a_batch.shape[0] == r_batch.shape[0]
    v_batch = critic.predict(s_batch)
    R_batch = np.zeros_like(r_batch, dtype=np.float32)

    R_batch[-1, 0] = 0.0 if terminal else float(v_batch[-1, 0])
    for t in reversed(range(len(r_batch) - 1)):
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
    entropy = 0.0
    for p in np.asarray(x, dtype=np.float32).reshape(-1):
        if 0.0 < p < 1.0:
            entropy -= float(p) * float(np.log(p))
    return float(entropy)


def build_summaries():
    return None, ["TD_loss", "Eps_total_reward", "Avg_entropy"]


def _to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=dtype, device=device)


def _torch_load(path, map_location):
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
    if isinstance(ckpt, (list, tuple)) and len(ckpt) >= 2:
        return ckpt[0], ckpt[1]
    if isinstance(ckpt, dict):
        if "actor" in ckpt:
            return ckpt["actor"], ckpt.get("critic")
        if "actor_state_dict" in ckpt:
            return ckpt["actor_state_dict"], ckpt.get("critic_state_dict")
        return ckpt, None
    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")
