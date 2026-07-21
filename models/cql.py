from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.dt import ObservationEncoder

OBS_KEYS = ("screens", "health", "level", "badges", "events", "map", "recent_actions")


def _get(config, key, default=None):
    cur = config
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


class PokemonTransitionDataset(Dataset):
    def __init__(self, dataset_dir, max_trajectories=None):
        self.dataset_dir = Path(dataset_dir)
        self.files = sorted(self.dataset_dir.glob("*.npz"))
        if max_trajectories is not None:
            self.files = self.files[: int(max_trajectories)]
        if not self.files:
            raise FileNotFoundError(f"No .npz trajectories found in {self.dataset_dir}")

        self.lengths = []
        for path in self.files:
            with np.load(path) as data:
                self.lengths.append(int(data["steps"]))
        self.lengths = np.asarray(self.lengths, dtype=np.int64)
        self.sample_probs = self.lengths / self.lengths.sum()

    def __len__(self):
        return int(self.lengths.sum())

    def _load_obs(self, data, prefix, index):
        obs = {}
        for key in OBS_KEYS:
            value = data[f"{prefix}_{key}"][index]
            if key in {"screens", "map"}:
                value = value.astype(np.float32) / 255.0
            elif key == "recent_actions":
                value = value.astype(np.int64)
            else:
                value = value.astype(np.float32)
            obs[f"{prefix}_{key}"] = torch.from_numpy(value)
        return obs

    def __getitem__(self, _):
        traj_idx = int(np.random.choice(len(self.files), p=self.sample_probs))
        path = self.files[traj_idx]

        with np.load(path) as data:
            traj_len = int(data["steps"])
            index = random.randrange(traj_len)

            batch = self._load_obs(data, "obs", index)
            batch.update(self._load_obs(data, "next_obs", index))
            batch["actions"] = torch.tensor(int(data["actions"][index]), dtype=torch.long)
            batch["rewards"] = torch.tensor(float(data["rewards"][index]), dtype=torch.float32)
            if "dones" in data:
                done = bool(data["dones"][index])
            else:
                done = bool(data["terminated"][index]) or bool(data["truncated"][index])
            batch["dones"] = torch.tensor(done, dtype=torch.float32)
        return batch


class CQLDQN(nn.Module):
    def __init__(self, action_dim=7, hidden_size=128, event_dim=2488):
        super().__init__()
        self.action_dim = int(action_dim)
        self.obs_encoder = ObservationEncoder(
            hidden_size=hidden_size,
            event_dim=event_dim,
            action_vocab_size=action_dim,
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def _as_sequence_batch(self, batch, prefix):
        return {
            f"obs_{key}": batch[f"{prefix}_{key}"].unsqueeze(1)
            for key in OBS_KEYS
        }

    def forward(self, batch, prefix="obs"):
        encoded = self.obs_encoder(self._as_sequence_batch(batch, prefix)).squeeze(1)
        return self.q_head(encoded)

    @staticmethod
    def _resolve_device(device):
        if device is None or device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested

    @staticmethod
    def _model_kwargs_from_checkpoint(checkpoint):
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        model_config = config.get("model", {}) if isinstance(config, dict) else {}

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        hidden_size = int(model_config.get("hidden_size", 128))
        action_dim = int(model_config.get("action_dim", 7))
        event_dim = int(model_config.get("event_dim", 2488))

        if isinstance(state_dict, dict):
            q_weight = state_dict.get("q_head.2.weight")
            if q_weight is not None:
                action_dim = int(q_weight.shape[0])
            hidden_weight = state_dict.get("q_head.0.weight")
            if hidden_weight is not None:
                hidden_size = int(hidden_weight.shape[0])
            vector_weight = state_dict.get("obs_encoder.vector_encoder.0.weight")
            if vector_weight is not None:
                vector_in = int(vector_weight.shape[1])
                event_dim = vector_in - (1 + 8 + 8 + hidden_size)

        return {
            "action_dim": action_dim,
            "hidden_size": hidden_size,
            "event_dim": event_dim,
        }

    @classmethod
    def load(cls, path, env=None, custom_objects=None, device="auto", **kwargs):
        del custom_objects
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_kwargs = cls._model_kwargs_from_checkpoint(checkpoint)
        model_kwargs.update(kwargs)

        model = cls(**model_kwargs)
        model.load_state_dict(state_dict)
        model.device = cls._resolve_device(device)
        model.to(model.device)
        model.eval()
        model.env = env
        model.checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        return model

    def _prepare_obs_value(self, key, value):
        if not torch.is_tensor(value):
            value = torch.from_numpy(np.asarray(value))
        if key in {"screens", "map"}:
            value = value.float()
            if value.numel() > 0 and value.max() > 1.0:
                value = value / 255.0
        elif key == "recent_actions":
            value = value.long()
        else:
            value = value.float()
        return value

    @torch.no_grad()
    def act(self, obs, device="cpu", deterministic=True):
        batch = {}
        device = self._resolve_device(device)
        for key in OBS_KEYS:
            value = self._prepare_obs_value(key, obs[key])
            batch[f"obs_{key}"] = value.unsqueeze(0).to(device)
        q_values = self.forward(batch, prefix="obs")
        print(f"q_values: {q_values.detach().cpu().numpy()}")

        if deterministic==False:
            # Epsilon-greedy exploration
            epsilon = 0.1  # You can adjust this value as needed
            if random.random() < epsilon:
                return random.randint(0, self.action_dim - 1)
            
        return int(q_values.argmax(dim=-1).item())
    
def move_batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def save_checkpoint(path, model, target_model, optimizer, step, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_model_state_dict": target_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )


def soft_update(target_model, model, tau):
    with torch.no_grad():
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.mul_(1.0 - tau).add_(param, alpha=tau)


def compute_cql_loss(model, target_model, batch, config):
    gamma = float(_get(config, "algorithm.gamma", 0.99))
    cql_alpha = float(_get(config, "algorithm.cql_alpha", 1.0))
    cql_temperature = float(_get(config, "algorithm.cql_temperature", 1.0))
    double_q = bool(_get(config, "algorithm.double_q", True))
    loss_type = _get(config, "algorithm.bellman_loss", "huber")

    q_values = model(batch, prefix="obs")
    actions = batch["actions"].long().clamp(0, model.action_dim - 1).unsqueeze(-1)
    q_action = q_values.gather(1, actions).squeeze(-1)

    with torch.no_grad():
        next_target_q_values = target_model(batch, prefix="next_obs")
        if double_q:
            next_actions = model(batch, prefix="next_obs").argmax(dim=-1, keepdim=True)
            next_q = next_target_q_values.gather(1, next_actions).squeeze(-1)
        else:
            next_q = next_target_q_values.max(dim=-1).values
        target_q = batch["rewards"] + gamma * (1.0 - batch["dones"]) * next_q

    if loss_type == "mse":
        bellman_loss = F.mse_loss(q_action, target_q)
    else:
        bellman_loss = F.smooth_l1_loss(q_action, target_q)

    conservative_q = torch.logsumexp(q_values / cql_temperature, dim=-1) * cql_temperature
    cql_loss = (conservative_q - q_action).mean()
    total_loss = bellman_loss + cql_alpha * cql_loss
    return total_loss, bellman_loss, cql_loss, q_action.mean(), target_q.mean()


def train(config):
    seed = int(_get(config, "training.seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_name = _get(config, "training.device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")

    dataset = PokemonTransitionDataset(
        dataset_dir=_get(config, "dataset.path", "offline_trajectories"),
        max_trajectories=_get(config, "dataset.max_trajectories", None),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(_get(config, "training.batch_size", 32)),
        shuffle=True,
        num_workers=int(_get(config, "training.num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = CQLDQN(
        action_dim=int(_get(config, "model.action_dim", 7)),
        hidden_size=int(_get(config, "model.hidden_size", 128)),
        event_dim=int(_get(config, "model.event_dim", 2488)),
    ).to(device)
    target_model = CQLDQN(
        action_dim=int(_get(config, "model.action_dim", 7)),
        hidden_size=int(_get(config, "model.hidden_size", 128)),
        event_dim=int(_get(config, "model.event_dim", 2488)),
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(_get(config, "optimizer.lr", 0.0001)),
        weight_decay=float(_get(config, "optimizer.weight_decay", 0.0001)),
        betas=(
            float(_get(config, "optimizer.beta1", 0.9)),
            float(_get(config, "optimizer.beta2", 0.999)),
        ),
    )

    max_steps = int(_get(config, "training.max_steps", 10000))
    log_interval = int(_get(config, "training.log_interval", 50))
    save_interval = int(_get(config, "training.save_interval", 1000))
    checkpoint_dir = Path(_get(config, "training.checkpoint_dir", "checkpoints/cql"))
    grad_clip = float(_get(config, "training.grad_clip", 10.0))
    target_update_interval = int(_get(config, "algorithm.target_update_interval", 1000))
    tau = float(_get(config, "algorithm.tau", 1.0))

    model.train()
    step = 0
    while step < max_steps:
        for batch in loader:
            step += 1
            batch = move_batch_to_device(batch, device)
            total_loss, bellman_loss, cql_loss, q_mean, target_q_mean = compute_cql_loss(
                model,
                target_model,
                batch,
                config,
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if target_update_interval > 0 and step % target_update_interval == 0:
                soft_update(target_model, model, tau)

            if step == 1 or step % log_interval == 0:
                print(
                    f"step {step:06d} | loss {total_loss.item():.4f} "
                    f"| bellman {bellman_loss.item():.4f} | cql {cql_loss.item():.4f} "
                    f"| q {q_mean.item():.4f} | target_q {target_q_mean.item():.4f} | device {device}"
                )
            if save_interval > 0 and step % save_interval == 0:
                save_checkpoint(checkpoint_dir / f"cql_step_{step}.pt", model, target_model, optimizer, step, config)
            if step >= max_steps:
                break

    final_path = checkpoint_dir / "cql_final.pt"
    save_checkpoint(final_path, model, target_model, optimizer, step, config)
    print(f"saved final checkpoint: {final_path}")
    return model
