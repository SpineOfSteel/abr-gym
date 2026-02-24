import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

GAMMA = 0.99
A_DIM = 6
S_INFO = 6   # ABR state rows used by Pensieve-style server/env
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6
ACTION_EPS = 1e-6


def _to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    arr = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _dbg(enabled, *msg):
    if enabled:
        print("[a3c_torch2]", *msg)


def _check_finite(name, x, enabled=False):
    if isinstance(x, torch.Tensor):
        ok = bool(torch.isfinite(x).all().item())
    else:
        ok = bool(np.isfinite(np.asarray(x)).all())
    if not ok:
        _dbg(enabled, f"Non-finite detected: {name}")
    return ok


def _torch_load(path, map_location):
    """Compatible with torch versions before/after weights_only kwarg."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _cpu_state_dict(sd):
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


def _looks_like_state_dict(obj):
    return isinstance(obj, dict) and len(obj) > 0 and all(isinstance(k, str) for k in obj.keys())


def _parse_checkpoint(ckpt):
    """
    Accepts:
      - [actor_state_dict, critic_state_dict]
      - (actor_state_dict, critic_state_dict)
      - {"actor": ..., "critic": ...}
      - {"actor_state_dict": ..., "critic_state_dict": ...}
      - actor-only state_dict (critic omitted)
    Returns: (actor_sd, critic_sd_or_none)
    """
    if isinstance(ckpt, (list, tuple)) and len(ckpt) >= 2 and _looks_like_state_dict(ckpt[0]):
        return ckpt[0], ckpt[1]

    if isinstance(ckpt, dict):
        if _looks_like_state_dict(ckpt.get("actor")) and _looks_like_state_dict(ckpt.get("critic")):
            return ckpt["actor"], ckpt["critic"]
        if _looks_like_state_dict(ckpt.get("actor_state_dict")) and _looks_like_state_dict(ckpt.get("critic_state_dict")):
            return ckpt["actor_state_dict"], ckpt["critic_state_dict"]
        # plain actor-only state dict
        if _looks_like_state_dict(ckpt):
            return ckpt, None

    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")


class _A3CBody(nn.Module):
    """
    Shared feature extractor similar to original Pensieve tflearn topology:
      - FC on rows 0,1,5 (latest scalar values)
      - Conv1D on rows 2,3,4 (history windows)
      - concat -> FC(128)
    Input shape: [B, S_INFO, S_LEN]
    """

    def __init__(self, state_dim, action_dim: int, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.s_info, self.s_len = int(state_dim[0]), int(state_dim[1])
        self.action_dim = int(action_dim)

        if self.s_info < 6:
            raise ValueError(f"S_INFO must be >= 6, got {self.s_info}")
        if self.s_len < 4:
            raise ValueError(f"S_LEN must be >= 4 for conv kernel=4, got {self.s_len}")
        if self.action_dim < 4:
            raise ValueError(f"A_DIM must be >= 4 for conv kernel=4 on row4, got {self.action_dim}")

        # Scalar branches (latest value only)
        self.fc0 = nn.Linear(1, 128)
        self.fc1 = nn.Linear(1, 128)
        self.fc5 = nn.Linear(1, 128)

        # Temporal branches
        self.conv2 = nn.Conv1d(1, 128, kernel_size=4)  # row 2, full S_LEN history
        self.conv3 = nn.Conv1d(1, 128, kernel_size=4)  # row 3, full S_LEN history
        self.conv4 = nn.Conv1d(1, 128, kernel_size=4)  # row 4, first A_DIM entries

        conv23_len = self.s_len - 4 + 1
        conv4_len = self.action_dim - 4 + 1
        merged_dim = 128 + 128 + (128 * conv23_len) + (128 * conv23_len) + (128 * conv4_len) + 128
        self.fc_merge = nn.Linear(merged_dim, 128)

        _dbg(self.verbose, f"Body init s_dim={state_dim}, a_dim={self.action_dim}, merged_dim={merged_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, S_INFO, S_LEN], got {tuple(x.shape)}")
        if x.shape[1] < 6:
            raise ValueError(f"Expected S_INFO>=6, got shape {tuple(x.shape)}")

        b0 = torch.relu(self.fc0(x[:, 0:1, -1]))
        b1 = torch.relu(self.fc1(x[:, 1:2, -1]))
        b5 = torch.relu(self.fc5(x[:, 5:6, -1]))

        b2 = torch.relu(self.conv2(x[:, 2:3, :]))
        b3 = torch.relu(self.conv3(x[:, 3:4, :]))
        b4 = torch.relu(self.conv4(x[:, 4:5, : self.action_dim]))

        b2f = torch.flatten(b2, start_dim=1)
        b3f = torch.flatten(b3, start_dim=1)
        b4f = torch.flatten(b4, start_dim=1)

        merged = torch.cat([b0, b1, b2f, b3f, b4f, b5], dim=1)
        out = torch.relu(self.fc_merge(merged))

        if self.verbose:
            _dbg(True, f"forward x={tuple(x.shape)} merged={tuple(merged.shape)} out={tuple(out.shape)}")
        return out


class _ActorModule(nn.Module):
    def __init__(self, state_dim, action_dim: int, verbose=False):
        super().__init__()
        self.body = _A3CBody(state_dim, action_dim, verbose=verbose)
        self.actor_head = nn.Linear(128, action_dim)

    def forward(self, x):
        z = self.body(x)
        logits = self.actor_head(z)
        probs = torch.softmax(logits, dim=1)
        probs = torch.clamp(probs, ACTION_EPS, 1.0 - ACTION_EPS)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


class _CriticModule(nn.Module):
    def __init__(self, state_dim, action_dim: int, verbose=False):
        super().__init__()
        self.body = _A3CBody(state_dim, action_dim, verbose=verbose)
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

        self.model = _ActorModule(state_dim=self.s_dim, action_dim=self.a_dim, verbose=verbose).to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr_rate)
        _dbg(self.verbose, f"Actor init device={self.device} s_dim={self.s_dim} a_dim={self.a_dim}")

    def _actor_loss(self, inputs, acts, act_grad_weights):
        x = _to_tensor(inputs, self.device)
        acts_t = _to_tensor(acts, self.device)
        w_t = _to_tensor(act_grad_weights, self.device)

        if x.dim() != 3:
            raise ValueError(f"Actor inputs must be [N,S_INFO,S_LEN], got {tuple(x.shape)}")
        if acts_t.dim() != 2:
            acts_t = acts_t.view(-1, self.a_dim)
        if w_t.dim() == 1:
            w_t = w_t.view(-1, 1)

        probs = self.model(x)
        selected_probs = torch.sum(probs * acts_t, dim=1, keepdim=True).clamp_min(ENTROPY_EPS)
        policy_term = torch.sum(torch.log(selected_probs) * (-w_t))
        entropy_term = ENTROPY_WEIGHT * torch.sum(probs * torch.log(probs.clamp_min(ENTROPY_EPS)))
        loss = policy_term + entropy_term

        if self.verbose:
            _dbg(True, f"Actor loss={float(loss.detach().cpu()):.6f} prob[min,max]=({float(probs.min()):.4e},{float(probs.max()):.4e})")
            _check_finite("actor_probs", probs, True)
            _check_finite("actor_loss", loss, True)
        return loss

    def train(self, inputs, acts, act_grad_weights):
        self.model.train()
        loss = self._actor_loss(inputs, acts, act_grad_weights)
        self.optimizer.zero_grad()
        loss.backward()
        if self.verbose:
            total_norm_sq = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    n = float(p.grad.data.norm(2).item())
                    total_norm_sq += n * n
            _dbg(True, f"Actor grad_norm={(total_norm_sq ** 0.5):.6f}")
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def predict(self, inputs):
        self.model.eval()
        x = _to_tensor(inputs, self.device)
        if x.dim() == 2:  # allow [S_INFO,S_LEN]
            x = x.unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
        arr = out.detach().cpu().numpy()
        if self.verbose:
            _dbg(True, f"Actor predict in={tuple(x.shape)} out={arr.shape}")
        return arr

    def get_gradients(self, inputs, acts, act_grad_weights):
        self.model.train()
        loss = self._actor_loss(inputs, acts, act_grad_weights)
        self.optimizer.zero_grad()
        loss.backward()
        grads = []
        for p in self.model.parameters():
            grads.append(None if p.grad is None else p.grad.detach().cpu().numpy().copy())
        return grads

    def apply_gradients(self, actor_gradients):
        self.optimizer.zero_grad()
        for p, g in zip(self.model.parameters(), actor_gradients):
            if g is None:
                continue
            p.grad = torch.from_numpy(np.asarray(g, dtype=np.float32)).to(self.device)
        self.optimizer.step()

    def get_network_params(self):
        return [p.detach().cpu().numpy().copy() for p in self.model.parameters()]

    def set_network_params(self, input_network_params):
        with torch.no_grad():
            for p, src in zip(self.model.parameters(), input_network_params):
                p.copy_(torch.from_numpy(np.asarray(src, dtype=np.float32)).to(self.device))

    def save(self, path):
        torch.save(_cpu_state_dict(self.model.state_dict()), path)

    def load(self, path, map_location=None):
        state = _torch_load(path, map_location=map_location or self.device)
        if isinstance(state, (list, tuple)) or (isinstance(state, dict) and ("actor" in state or "actor_state_dict" in state)):
            actor_sd, _ = _parse_checkpoint(state)
            state = actor_sd
        self.model.load_state_dict(state)
        _dbg(self.verbose, f"Actor loaded from {path}")


class CriticNetwork:
    def __init__(self, state_dim, learning_rate, action_dim=A_DIM, device=None, verbose=False):
        self.s_dim = state_dim
        self.lr_rate = learning_rate
        self.a_dim = action_dim
        self.verbose = verbose
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = _CriticModule(state_dim=self.s_dim, action_dim=self.a_dim, verbose=verbose).to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr_rate)
        self.mse = nn.MSELoss()
        _dbg(self.verbose, f"Critic init device={self.device} s_dim={self.s_dim} a_dim={self.a_dim}")

    def _critic_loss(self, inputs, td_target):
        x = _to_tensor(inputs, self.device)
        y = _to_tensor(td_target, self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.view(-1, 1)
        out = self.model(x)
        loss = self.mse(out, y)
        if self.verbose:
            _check_finite("critic_pred", out, True)
            _check_finite("critic_target", y, True)
            _check_finite("critic_loss", loss, True)
        return loss

    def train(self, inputs, td_target):
        self.model.train()
        loss = self._critic_loss(inputs, td_target)
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
        arr = out.detach().cpu().numpy()
        if self.verbose:
            _dbg(True, f"Critic predict in={tuple(x.shape)} out={arr.shape}")
        return arr

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
        loss = self._critic_loss(inputs, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        grads = []
        for p in self.model.parameters():
            grads.append(None if p.grad is None else p.grad.detach().cpu().numpy().copy())
        return grads

    def apply_gradients(self, critic_gradients):
        self.optimizer.zero_grad()
        for p, g in zip(self.model.parameters(), critic_gradients):
            if g is None:
                continue
            p.grad = torch.from_numpy(np.asarray(g, dtype=np.float32)).to(self.device)
        self.optimizer.step()

    def get_network_params(self):
        return [p.detach().cpu().numpy().copy() for p in self.model.parameters()]

    def set_network_params(self, input_network_params):
        with torch.no_grad():
            for p, src in zip(self.model.parameters(), input_network_params):
                p.copy_(torch.from_numpy(np.asarray(src, dtype=np.float32)).to(self.device))

    def save(self, path):
        torch.save(_cpu_state_dict(self.model.state_dict()), path)

    def load(self, path, map_location=None):
        state = _torch_load(path, map_location=map_location or self.device)
        if isinstance(state, (list, tuple)) or (isinstance(state, dict) and ("critic" in state or "critic_state_dict" in state)):
            _, critic_sd = _parse_checkpoint(state)
            if critic_sd is None:
                raise ValueError("Checkpoint does not contain critic weights")
            state = critic_sd
        self.model.load_state_dict(state)
        _dbg(self.verbose, f"Critic loaded from {path}")


class Network:
    

    def __init__(self, state_dim, action_dim, learning_rate, critic_learning_rate=1e-3, device=None, verbose=False):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.c_lr_rate = critic_learning_rate
        self.verbose = verbose

        self.actor = ActorNetwork(state_dim, action_dim, learning_rate, device=device, verbose=verbose)
        self.critic = CriticNetwork(state_dim, critic_learning_rate, action_dim=action_dim, device=device, verbose=verbose)

        # compatibility field (older scripts print this)
        self._entropy_weight = ENTROPY_WEIGHT
        _dbg(self.verbose, f"Compat Network init s_dim={state_dim} a_dim={action_dim}")

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
        if self.verbose:
            _dbg(True, f"Compat predict probs={np.array2string(probs, precision=4)}")
            _check_finite("compat_probs", probs, True)
        return probs

    def train(self, s_batch, a_batch, p_batch, v_batch, epoch=None):
        # p_batch is old policy in PPO; ignored here (A3C-style update)
        s_batch = np.asarray(s_batch, dtype=np.float32)
        a_batch = np.asarray(a_batch, dtype=np.float32)
        R_batch = np.asarray(v_batch, dtype=np.float32).reshape(-1, 1)

        values = self.critic.predict(s_batch)
        adv = R_batch - values

        actor_loss = self.actor.train(s_batch, a_batch, adv)
        critic_loss, _ = self.critic.train(s_batch, R_batch)

        if self.verbose:
            _dbg(True, f"Compat train epoch={epoch} actor_loss={actor_loss:.6f} critic_loss={critic_loss:.6f} "
                       f"adv_mean={float(np.mean(adv)):.6f}")
        return actor_loss, critic_loss

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        """Return discounted returns (same shape behavior as ppo2.Network.compute_v())."""
        s_batch = np.asarray(s_batch, dtype=np.float32)
        r_batch = np.asarray(r_batch, dtype=np.float32).reshape(-1, 1)
        if len(r_batch) == 0:
            return []

        R_batch = np.zeros_like(r_batch, dtype=np.float32)
        if terminal:
            R_batch[-1, 0] = float(r_batch[-1, 0])
        else:
            val = self.critic.predict(s_batch)
            R_batch[-1, 0] = float(val[-1, 0])

        for t in reversed(range(len(r_batch) - 1)):
            R_batch[t, 0] = float(r_batch[t, 0]) + GAMMA * float(R_batch[t + 1, 0])

        return list(R_batch)

    def get_network_params(self):
        # format matches ppo2 style: [actor_state_dict, critic_state_dict]
        return [_cpu_state_dict(self.actor.model.state_dict()), _cpu_state_dict(self.critic.model.state_dict())]

    def set_network_params(self, input_network_params):
        actor_sd, critic_sd = _parse_checkpoint(input_network_params)
        self.actor.model.load_state_dict(actor_sd)
        if critic_sd is not None:
            self.critic.model.load_state_dict(critic_sd)
        elif self.verbose:
            _dbg(True, "set_network_params received actor-only checkpoint; critic unchanged")

    def load_model(self, nn_model):
        ckpt = _torch_load(nn_model, map_location=self.actor.device)
        actor_sd, critic_sd = _parse_checkpoint(ckpt)
        self.actor.model.load_state_dict(actor_sd)
        if critic_sd is not None:
            self.critic.model.load_state_dict(critic_sd)
        _dbg(self.verbose, f"Loaded checkpoint from {nn_model} (critic_loaded={critic_sd is not None})")

    def save_model(self, nn_model):
        torch.save(self.get_network_params(), nn_model)
        _dbg(self.verbose, f"Saved compat checkpoint to {nn_model}")


def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic, verbose=False):
    s_batch = np.asarray(s_batch, dtype=np.float32)
    a_batch = np.asarray(a_batch, dtype=np.float32)
    r_batch = np.asarray(r_batch, dtype=np.float32).reshape(-1, 1)

    assert s_batch.shape[0] == a_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]

    v_batch = critic.predict(s_batch)  # [N,1]
    R_batch = np.zeros_like(r_batch, dtype=np.float32)

    if terminal:
        R_batch[-1, 0] = 0.0
    else:
        R_batch[-1, 0] = float(v_batch[-1, 0])

    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = float(r_batch[t, 0]) + GAMMA * float(R_batch[t + 1, 0])

    td_batch = R_batch - v_batch

    if verbose:
        _dbg(True, f"compute_gradients N={ba_size} terminal={terminal} td_mean={float(np.mean(td_batch)):.6f}")

    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
    critic_gradients = critic.get_gradients(s_batch, R_batch)
    return actor_gradients, critic_gradients, td_batch


def discount(x, gamma):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        x = x.reshape(-1)
    out = np.zeros(len(x), dtype=np.float32)
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
    # TensorFlow summaries are not used in this PyTorch version.
    return None, ["TD_loss", "Eps_total_reward", "Avg_entropy"]
