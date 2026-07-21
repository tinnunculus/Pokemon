from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

OBS_KEYS = ("screens", "health", "level", "badges", "events", "map", "recent_actions")

def _get(config, key, default=None):
    cur = config
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

class PokemonTrajectoryDataset(Dataset):
    def __init__(self, dataset_dir, context_len, rtg_scale=1.0, max_trajectories=None):
        self.dataset_dir = Path(dataset_dir)
        self.context_len = int(context_len)
        self.rtg_scale = float(rtg_scale)
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

    def _empty_obs(self):
        return {
            "screens": np.zeros((self.context_len, 72, 80, 3), dtype=np.float32),
            "health": np.zeros((self.context_len, 1), dtype=np.float32),
            "level": np.zeros((self.context_len, 8), dtype=np.float32),
            "badges": np.zeros((self.context_len, 8), dtype=np.float32),
            "events": np.zeros((self.context_len, 2488), dtype=np.float32),
            "map": np.zeros((self.context_len, 48, 48, 1), dtype=np.float32),
            "recent_actions": np.zeros((self.context_len, 3), dtype=np.int64),
        }

    def __getitem__(self, _):
        traj_idx = int(np.random.choice(len(self.files), p=self.sample_probs))
        path = self.files[traj_idx]

        with np.load(path) as data:
            traj_len = int(data["steps"])
            end = random.randint(1, traj_len)
            start = max(0, end - self.context_len)
            seq_len = end - start
            pad = self.context_len - seq_len

            obs = self._empty_obs()
            obs["screens"][pad:] = data["obs_screens"][start:end].astype(np.float32) / 255.0
            obs["health"][pad:] = data["obs_health"][start:end].astype(np.float32)
            obs["level"][pad:] = data["obs_level"][start:end].astype(np.float32)
            obs["badges"][pad:] = data["obs_badges"][start:end].astype(np.float32)
            obs["events"][pad:] = data["obs_events"][start:end].astype(np.float32)
            obs["map"][pad:] = data["obs_map"][start:end].astype(np.float32) / 255.0
            obs["recent_actions"][pad:] = data["obs_recent_actions"][start:end].astype(np.int64)

            actions = np.zeros((self.context_len,), dtype=np.int64)
            rewards = np.zeros((self.context_len,), dtype=np.float32)
            timesteps = np.zeros((self.context_len,), dtype=np.int64)
            mask = np.zeros((self.context_len,), dtype=np.float32)
            rtg = np.zeros((self.context_len,), dtype=np.float32)

            traj_actions = data["actions"][start:end].astype(np.int64)
            traj_rewards = data["rewards"].astype(np.float32)
            actions[pad:] = traj_actions
            rewards[pad:] = traj_rewards[start:end]
            timesteps[pad:] = np.arange(start, end, dtype=np.int64)
            mask[pad:] = 1.0

            future_returns = np.cumsum(traj_rewards[::-1])[::-1]
            rtg[pad:] = future_returns[start:end] / self.rtg_scale

        batch = {
            "actions": torch.from_numpy(actions),
            "rewards": torch.from_numpy(rewards),
            "returns_to_go": torch.from_numpy(rtg),
            "timesteps": torch.from_numpy(timesteps),
            "mask": torch.from_numpy(mask),
        }
        for key, value in obs.items():
            batch[f"obs_{key}"] = torch.from_numpy(value)
        return batch

class ObservationEncoder(nn.Module):
    def __init__(self, hidden_size, event_dim=2488, action_vocab_size=7):
        super().__init__()
        self.screen_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 8, hidden_size),
            nn.ReLU(),
        )
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, hidden_size),
            nn.ReLU(),
        )
        self.recent_action_embed = nn.Embedding(action_vocab_size, hidden_size)
        vector_dim = 1 + 8 + 8 + event_dim + hidden_size
        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Tanh(),
        )

    def forward(self, batch):
        screens = batch["obs_screens"].permute(0, 1, 4, 2, 3)
        maps = batch["obs_map"].permute(0, 1, 4, 2, 3)
        batch_size, context_len = screens.shape[:2]

        screen_emb = self.screen_encoder(screens.reshape(batch_size * context_len, 3, 72, 80))
        map_emb = self.map_encoder(maps.reshape(batch_size * context_len, 1, 48, 48))

        recent_actions = batch["obs_recent_actions"].long().clamp_min(0)
        recent_emb = self.recent_action_embed(recent_actions).mean(dim=2)
        vector = torch.cat(
            [
                batch["obs_health"].float(),
                batch["obs_level"].float(),
                batch["obs_badges"].float(),
                batch["obs_events"].float(),
                recent_emb,
            ],
            dim=-1,
        )
        vector_emb = self.vector_encoder(vector.reshape(batch_size * context_len, -1))
        state_emb = torch.cat([screen_emb, map_emb, vector_emb], dim=-1)
        return self.out(state_emb).reshape(batch_size, context_len, -1)

class DecisionTransformer(nn.Module):
    def __init__(
        self,
        action_dim=7,
        hidden_size=128,
        context_len=30,
        n_layer=4,
        n_head=4,
        dropout=0.1,
        max_timestep=50000,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.context_len = int(context_len)

        self.obs_encoder = ObservationEncoder(hidden_size, action_vocab_size=action_dim)
        self.action_embed = nn.Embedding(action_dim, hidden_size)
        self.rtg_embed = nn.Linear(1, hidden_size)
        self.timestep_embed = nn.Embedding(max_timestep + 1, hidden_size)
        self.token_ln = nn.LayerNorm(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.action_head = nn.Linear(hidden_size, action_dim)

    def forward(self, batch):
        batch_size, context_len = batch["actions"].shape
        timesteps = batch["timesteps"].clamp(0, self.timestep_embed.num_embeddings - 1)
        time_emb = self.timestep_embed(timesteps)

        state_tokens = self.obs_encoder(batch) + time_emb
        action_tokens = self.action_embed(batch["actions"].long().clamp(0, self.action_dim - 1)) + time_emb
        rtg_tokens = self.rtg_embed(batch["returns_to_go"].float().unsqueeze(-1)) + time_emb

        stacked = torch.stack([rtg_tokens, state_tokens, action_tokens], dim=2)
        tokens = self.token_ln(stacked.reshape(batch_size, 3 * context_len, self.hidden_size))

        causal_mask = torch.triu(
            torch.ones(3 * context_len, 3 * context_len, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        attention_mask = batch["mask"].repeat_interleave(3, dim=1) == 0
        outputs = self.transformer(tokens, mask=causal_mask, src_key_padding_mask=attention_mask)
        state_outputs = outputs[:, 1::3]
        return self.action_head(state_outputs)

def move_batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}

def save_checkpoint(path, model, optimizer, step, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )

def train(config):
    seed = int(_get(config, "training.seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_name = _get(config, "training.device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")

    context_len = int(_get(config, "model.context_len", 30))
    dataset = PokemonTrajectoryDataset(
        dataset_dir=_get(config, "dataset.path", "offline_trajectories"),
        context_len=context_len,
        rtg_scale=float(_get(config, "dataset.rtg_scale", 1000.0)),
        max_trajectories=_get(config, "dataset.max_trajectories", None),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(_get(config, "training.batch_size", 8)),
        shuffle=True,
        num_workers=int(_get(config, "training.num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = DecisionTransformer(
        action_dim=int(_get(config, "model.action_dim", 7)),
        hidden_size=int(_get(config, "model.hidden_size", 128)),
        context_len=context_len,
        n_layer=int(_get(config, "model.n_layer", 4)),
        n_head=int(_get(config, "model.n_head", 4)),
        dropout=float(_get(config, "model.dropout", 0.1)),
        max_timestep=int(_get(config, "model.max_timestep", 50000)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(_get(config, "optimizer.lr", 0.0001)),
        weight_decay=float(_get(config, "optimizer.weight_decay", 0.0001)),
        betas=(
            float(_get(config, "optimizer.beta1", 0.9)),
            float(_get(config, "optimizer.beta2", 0.95)),
        ),
    )

    max_steps = int(_get(config, "training.max_steps", 10000))
    log_interval = int(_get(config, "training.log_interval", 50))
    save_interval = int(_get(config, "training.save_interval", 1000))
    checkpoint_dir = Path(_get(config, "training.checkpoint_dir", "checkpoints/dt"))
    grad_clip = float(_get(config, "training.grad_clip", 1.0))

    model.train()
    step = 0
    while step < max_steps:
        for batch in loader:
            step += 1
            batch = move_batch_to_device(batch, device)
            logits = model(batch)

            loss = F.cross_entropy(
                logits.reshape(-1, model.action_dim),
                batch["actions"].reshape(-1),
                reduction="none",
            )
            loss = (loss * batch["mask"].reshape(-1)).sum() / batch["mask"].sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if step == 1 or step % log_interval == 0:
                print(f"step {step:06d} | loss {loss.item():.4f} | device {device}")
            if save_interval > 0 and step % save_interval == 0:
                save_checkpoint(checkpoint_dir / f"dt_step_{step}.pt", model, optimizer, step, config)
            if step >= max_steps:
                break

    final_path = checkpoint_dir / "dt_final.pt"
    save_checkpoint(final_path, model, optimizer, step, config)
    print(f"saved final checkpoint: {final_path}")
    return model
 