import copy
import random
import time
from pathlib import Path
from tqdm.auto import tqdm
from functools import partial
import time
import os

from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update
from jaxrl_m.networks import Policy, Critic, ensemblize, DiscretePolicy
from utils import CsvLogger


import flax
import flax.linen as nn
from flax.core import freeze, unfreeze
import ml_collections
from . import iql
from .special_networks import Representation, HierarchicalActorCritic, RelativeRepresentation, MonolithicVF

try:
    tree_map = jax.tree.map
except AttributeError:  # pragma: no cover
    tree_map = jax.tree_util.tree_map

def _get(config, key, default=None):
    cur = config
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

OBS_KEYS = ("screens", "health", "level", "badges", "events", "map", "recent_actions")


def flatten_observation(obs):
    parts = []
    for key in OBS_KEYS:
        value = obs[key].astype(np.float32)
        if key in {"screens", "map"}:
            value = value / 255.0
        parts.append(value.ravel())
    return np.concatenate(parts, axis=-1)


def flatten_observation_batch(obs):
    batch_size = next(iter(obs.values())).shape[0]
    parts = []
    for key in OBS_KEYS:
        value = obs[key].astype(np.float32)
        if key in {"screens", "map"}:
            value = value / 255.0
        parts.append(value.reshape(batch_size, -1))
    return np.concatenate(parts, axis=-1)


class HiQLDataset:
    """Goal-conditioned HIQL batches backed by a directory of trajectories."""

    def __init__(
        self,
        dataset_dir,
        max_trajectories=None,
        p_randomgoal=0.3,
        p_trajgoal=0.5,
        p_currgoal=0.2,
        geom_sample=0,
        discount=0.99,
        way_steps=0,
        high_p_randomgoal=0.0,
        reward_scale=1.0,
        reward_shift=0.0,
        terminal=False,
        trajectories_per_batch=None,
    ):
        self.files = sorted(Path(dataset_dir).glob("*.npz"))
        if max_trajectories is not None:
            self.files = self.files[:int(max_trajectories)]
        if not self.files:
            raise FileNotFoundError(f"No .npz trajectories found in {dataset_dir}")

        lengths = []    # 각 trajectory별 step 수를 저장하는 리스트
        for path in self.files:
            with np.load(path) as trajectory:
                lengths.append(int(trajectory["steps"]))
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.terminal_locs = np.cumsum(self.lengths) - 1    # 전체 데이터셋에서 각 trajectory의 마지막 step index를 저장하는 배열

        self.p_randomgoal = float(p_randomgoal)
        self.p_trajgoal = float(p_trajgoal)
        self.p_currgoal = float(p_currgoal)
        self.geom_sample = bool(geom_sample)
        self.discount = float(discount)
        self.way_steps = int(way_steps)
        self.high_p_randomgoal = float(high_p_randomgoal)
        self.reward_scale = float(reward_scale)
        self.reward_shift = float(reward_shift)
        self.terminal = bool(terminal)
        self.trajectories_per_batch = (
            None
            if trajectories_per_batch is None
            else int(trajectories_per_batch)
        )
        self._data_cache = {}

        if not np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0):
            raise ValueError("Goal sampling probabilities must sum to 1")
        if (
            self.trajectories_per_batch is not None
            and self.trajectories_per_batch < 1
        ):
            raise ValueError("trajectories_per_batch must be at least 1")

    def __len__(self):
        return int(self.terminal_locs[-1] + 1)   # 전체 step 수 (모든 trajectory의 step 수 합)

    def _load(self, trajectory_index):
        trajectory_index = int(trajectory_index)
        if trajectory_index not in self._data_cache:
            self._data_cache[trajectory_index] = np.load(self.files[trajectory_index])
        return self._data_cache[trajectory_index]

    def _locate(self, indices):
        trajectory_indices = np.searchsorted(self.terminal_locs, indices)
        starts = self.terminal_locs[trajectory_indices] - self.lengths[trajectory_indices] + 1
        return trajectory_indices, indices - starts

    def _sample_random_indices(self, size):
        """Sample globally uniform steps using a small trajectory pool.

        Selecting trajectories proportional to their lengths and then selecting
        local steps uniformly preserves the marginal distribution of the old
        ``np.random.randint(len(self))`` sampler. Limiting the pool only adds
        correlation within a batch, which prevents compressed trajectory files
        from being opened and decompressed hundreds of times per update.
        """
        if self.trajectories_per_batch is None:
            return np.random.randint(len(self), size=size)

        pool_size = min(self.trajectories_per_batch, len(self.files))
        trajectory_pool = np.random.choice(
            len(self.files),
            size=pool_size,
            replace=True,
            p=self.lengths / len(self),
        )
        selected_trajectories = trajectory_pool[
            np.random.randint(pool_size, size=size)
        ]
        local_indices = (
            np.random.rand(size) * self.lengths[selected_trajectories]
        ).astype(np.int64)
        starts = (
            self.terminal_locs[selected_trajectories]
            - self.lengths[selected_trajectories]
            + 1
        )
        return starts + local_indices

    def _gather_batch(self, observation_requests, action_indices=None):
        """Gather all fields for a batch while reading each NPZ member once.

        ``observation_requests`` maps an output name to ``(indices, prefix)``.
        Requests that use the same prefix (for example observations and goals)
        are merged per trajectory before the corresponding NPZ arrays are read.
        """
        requests = {}
        trajectory_parts = []
        for name, (indices, prefix) in observation_requests.items():
            indices = np.asarray(indices, dtype=np.int64)
            trajectory_indices, local_indices = self._locate(indices)
            requests[name] = (indices, prefix, trajectory_indices, local_indices)
            trajectory_parts.append(trajectory_indices)

        action_request = None
        if action_indices is not None:
            action_indices = np.asarray(action_indices, dtype=np.int64)
            action_trajectories, action_locals = self._locate(action_indices)
            action_request = (action_indices, action_trajectories, action_locals)
            trajectory_parts.append(action_trajectories)

        results = {name: None for name in requests}
        if action_request is not None:
            results["actions"] = np.empty(len(action_indices), dtype=np.int64)

        if not trajectory_parts or not any(len(part) for part in trajectory_parts):
            return results

        used_trajectories = np.unique(np.concatenate(trajectory_parts))
        for trajectory_index in used_trajectories:
            # All observation/goal/action requests for this trajectory share
            # this single archive lookup.
            trajectory = self._load(trajectory_index)

            requests_by_prefix = {}
            for name, (_, prefix, trajectory_indices, local_indices) in requests.items():
                positions = np.flatnonzero(trajectory_indices == trajectory_index)
                if len(positions):
                    requests_by_prefix.setdefault(prefix, []).append(
                        (name, positions, local_indices[positions])
                    )

            for prefix, prefix_requests in requests_by_prefix.items():
                all_local_indices = np.concatenate(
                    [local_indices for _, _, local_indices in prefix_requests]
                )
                unique_local_indices, inverse = np.unique(
                    all_local_indices, return_inverse=True
                )

                # An NPZ member is decompressed when it is accessed. Reading
                # every observation member here avoids decompressing it again
                # for observations, goals, low_goals, and high_goals.
                observations = {
                    key: np.asarray(trajectory[f"{prefix}_{key}"])[
                        unique_local_indices
                    ]
                    for key in OBS_KEYS
                }
                values = flatten_observation_batch(observations)

                offset = 0
                for name, positions, local_indices in prefix_requests:
                    count = len(local_indices)
                    if results[name] is None:
                        results[name] = np.empty(
                            (len(requests[name][0]), values.shape[1]),
                            dtype=np.float32,
                        )
                    results[name][positions] = values[inverse[offset:offset + count]]
                    offset += count

            if action_request is not None:
                _, action_trajectories, action_locals = action_request
                positions = np.flatnonzero(action_trajectories == trajectory_index)
                if len(positions):
                    results["actions"][positions] = np.asarray(
                        trajectory["actions"]
                    )[action_locals[positions]]

        return results

    def _gather_observations(self, indices, prefix="obs"):
        return self._gather_batch(
            {"observations": (indices, prefix)}
        )["observations"]

    def _gather_actions(self, indices):
        return self._gather_batch({}, action_indices=indices)["actions"]

    def sample_goals(self, indices):
        batch_size = len(indices)

        # Goals from the same trajectory
        final_indices = self.terminal_locs[np.searchsorted(self.terminal_locs, indices)]   # 선택된 각 trajectory의 마지막 step index를 가져온다.

        # Random goals
        goals = self._sample_random_indices(batch_size)

        if self.geom_sample:
            offsets = np.ceil(
                np.log1p(-np.random.rand(batch_size)) / np.log(self.discount)
            ).astype(np.int64)
            trajectory_goals = np.minimum(indices + offsets, final_indices)
        else:
            distance = np.random.rand(batch_size)
            trajectory_goals = np.rint(
                np.minimum(indices + 1, final_indices) * distance
                + final_indices * (1.0 - distance)
            ).astype(np.int64)

        trajectory_probability = self.p_trajgoal / (1.0 - self.p_currgoal)
        goals = np.where(
            np.random.rand(batch_size) < trajectory_probability,
            trajectory_goals,
            goals,
        )
        return np.where(np.random.rand(batch_size) < self.p_currgoal, indices, goals)

    def sample(self, batch_size, indx=None):
        indices = (
            self._sample_random_indices(batch_size)
            if indx is None
            else np.asarray(indx, dtype=np.int64)
        )
        batch_size = len(indices)
        goal_indices = self.sample_goals(indices)
        final_indices = self.terminal_locs[np.searchsorted(self.terminal_locs, indices)]
        way_indices = np.minimum(indices + self.way_steps, final_indices)

        distance = np.random.rand(batch_size)
        high_trajectory_goals = np.rint(
            np.minimum(indices + 1, final_indices) * distance
            + final_indices * (1.0 - distance)
        ).astype(np.int64)
        # GCSDataset semantics: sample a distant future high-level goal, then
        # train its target at most way_steps ahead without passing that goal.
        high_trajectory_targets = np.minimum(
            indices + self.way_steps, high_trajectory_goals
        )
        random_high_goals = self._sample_random_indices(batch_size)
        use_random_high_goal = np.random.rand(batch_size) < self.high_p_randomgoal
        high_goal_indices = np.where(
            use_random_high_goal, random_high_goals, high_trajectory_goals
        )
        high_target_indices = np.where(
            use_random_high_goal, way_indices, high_trajectory_targets
        )

        success = (indices == goal_indices).astype(np.float32)
        masks = (
            1.0 - success
            if self.terminal
            else np.ones(batch_size, dtype=np.float32)
        )

        # Gather every indexed field together so each trajectory and each
        # compressed NPZ member is visited only once for this sample.
        batch = self._gather_batch(
            {
                "observations": (indices, "obs"),
                "next_observations": (indices, "next_obs"),
                "goals": (goal_indices, "obs"),
                "low_goals": (way_indices, "obs"),
                "high_goals": (high_goal_indices, "obs"),
                "high_targets": (high_target_indices, "obs"),
            },
            action_indices=indices,
        )
        batch.update({
            "rewards": success * self.reward_scale + self.reward_shift,
            "masks": masks,
        })
        return batch

def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)


def compute_actor_loss(agent, batch, network_params):
    if agent.config['use_waypoints']:  # Use waypoint states as goals (for hierarchical policies)
        cur_goals = batch['low_goals']
    else:  # Use randomized last observations as goals (for flat policies)
        cur_goals = batch['high_goals']
    v1, v2 = agent.network(batch['observations'], cur_goals, method='value')
    nv1, nv2 = agent.network(batch['next_observations'], cur_goals, method='value')
    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2

    adv = nv - v
    exp_a = jnp.exp(adv * agent.config['temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)

    if agent.config['use_waypoints']:
        goal_rep_grad = agent.config['policy_train_rep']
    else:
        goal_rep_grad = True
    dist = agent.network(batch['observations'], cur_goals, state_rep_grad=True, goal_rep_grad=goal_rep_grad, method='actor', params=network_params)
    log_probs = dist.log_prob(batch['actions'])
    actor_loss = -(exp_a * log_probs).mean()

    return actor_loss, {
        'actor_loss': actor_loss,
        'adv': adv.mean(),
        'bc_log_probs': log_probs.mean(),
        'adv_median': jnp.median(adv),
        'mse': jnp.mean((dist.mode() - batch['actions'])**2),
    }


def compute_high_actor_loss(agent, batch, network_params):
    cur_goals = batch['high_goals']
    v1, v2 = agent.network(batch['observations'], cur_goals, method='value')
    nv1, nv2 = agent.network(batch['high_targets'], cur_goals, method='value')
    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2

    adv = nv - v
    exp_a = jnp.exp(adv * agent.config['high_temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)

    dist = agent.network(batch['observations'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=True, method='high_actor', params=network_params)
    if agent.config['use_rep']:
        target = agent.network(targets=batch['high_targets'], bases=batch['observations'], method='value_goal_encoder')
    else:
        target = batch['high_targets'] - batch['observations']
    log_probs = dist.log_prob(target)
    actor_loss = -(exp_a * log_probs).mean()

    return actor_loss, {
        'high_actor_loss': actor_loss,
        'high_adv': adv.mean(),
        'high_bc_log_probs': log_probs.mean(),
        'high_adv_median': jnp.median(adv),
        'high_mse': jnp.mean((dist.mode() - target)**2),
        'high_scale': dist.scale_diag.mean(),
    }


def compute_value_loss(agent, batch, network_params):
    # masks are 0 if terminal, 1 otherwise
    batch['masks'] = 1.0 - batch['rewards']
    # rewards are 0 if terminal, -1 otherwise
    batch['rewards'] = batch['rewards'] - 1.0

    (next_v1, next_v2) = agent.network(batch['next_observations'], batch['goals'], method='target_value')
    next_v = jnp.minimum(next_v1, next_v2)
    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

    (v1_t, v2_t) = agent.network(batch['observations'], batch['goals'], method='target_value')
    v_t = (v1_t + v2_t) / 2
    adv = q - v_t

    q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1
    q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2
    (v1, v2) = agent.network(batch['observations'], batch['goals'], method='value', params=network_params)

    value_loss1 = expectile_loss(adv, q1 - v1, agent.config['pretrain_expectile']).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, agent.config['pretrain_expectile']).mean()
    value_loss = value_loss1 + value_loss2

    advantage = adv
    return value_loss, {
        'value_loss': value_loss,
        'v max': v1.max(),
        'v min': v1.min(),
        'v mean': v1.mean(),
        'abs adv mean': jnp.abs(advantage).mean(),
        'adv mean': advantage.mean(),
        'adv max': advantage.max(),
        'adv min': advantage.min(),
        'accept prob': (advantage >= 0).mean(),
    }


class JointTrainAgent(iql.IQLAgent):
    network: TrainState = None

    def pretrain_update(agent, pretrain_batch, seed=None, value_update=True, actor_update=True, high_actor_update=True):
        def loss_fn(network_params):
            info = {}

            # Value
            if value_update:
                value_loss, value_info = compute_value_loss(agent, pretrain_batch, network_params)
                for k, v in value_info.items():
                    info[f'value/{k}'] = v
            else:
                value_loss = 0.

            # Actor
            if actor_update:
                actor_loss, actor_info = compute_actor_loss(agent, pretrain_batch, network_params)
                for k, v in actor_info.items():
                    info[f'actor/{k}'] = v
            else:
                actor_loss = 0.

            # High Actor
            if high_actor_update and agent.config['use_waypoints']:
                high_actor_loss, high_actor_info = compute_high_actor_loss(agent, pretrain_batch, network_params)
                for k, v in high_actor_info.items():
                    info[f'high_actor/{k}'] = v
            else:
                high_actor_loss = 0.

            loss = value_loss + actor_loss + high_actor_loss

            return loss, info

        if value_update:
            new_target_params = tree_map(
                lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_value'], agent.network.params['networks_target_value']
            )

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)

        if value_update:
            params = unfreeze(new_network.params)
            params['networks_target_value'] = new_target_params
            new_network = new_network.replace(params=freeze(params))

        return agent.replace(network=new_network), info
    pretrain_update = jax.jit(pretrain_update, static_argnames=('value_update', 'actor_update', 'high_actor_update'))

    def sample_actions(agent,
                       observations: np.ndarray,
                       goals: np.ndarray,
                       *,
                       low_dim_goals: bool = False,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       discrete: int = 0,
                       num_samples: int = None) -> jnp.ndarray:
        dist = agent.network(observations, goals, low_dim_goals=low_dim_goals, temperature=temperature, method='actor')
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        if not discrete:
            actions = jnp.clip(actions, -1, 1)
        return actions
    sample_actions = jax.jit(sample_actions, static_argnames=('num_samples', 'low_dim_goals', 'discrete'))

    def sample_high_actions(agent,
                            observations: np.ndarray,
                            goals: np.ndarray,
                            *,
                            seed: PRNGKey,
                            temperature: float = 1.0,
                            num_samples: int = None) -> jnp.ndarray:
        dist = agent.network(observations, goals, temperature=temperature, method='high_actor')
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        return actions
    sample_high_actions = jax.jit(sample_high_actions, static_argnames=('num_samples',))

    @jax.jit
    def get_policy_rep(agent,
                       *,
                       targets: np.ndarray,
                       bases: np.ndarray = None,
                       ) -> jnp.ndarray:
        return agent.network(targets=targets, bases=bases, method='policy_goal_encoder')


def create_learner(
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        temperature: float = 1,
        high_temperature: float = 1,
        pretrain_expectile: float = 0.7,
        way_steps: int = 0,
        rep_dim: int = 10,
        use_rep: int = 0,
        policy_train_rep: float = 0,
        visual: int = 0,
        encoder: str = 'impala',
        discrete: int = 0,
        use_layer_norm: int = 0,
        rep_type: str = 'state',
        use_waypoints: int = 0,
        **kwargs):

        # print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, high_actor_key, critic_key, value_key = jax.random.split(rng, 5)

        value_state_encoder = None
        value_goal_encoder = None
        policy_state_encoder = None
        policy_goal_encoder = None
        high_policy_state_encoder = None
        high_policy_goal_encoder = None
        if visual:
            assert use_rep
            from jaxrl_m.vision import encoders

            visual_encoder = encoders[encoder]
            def make_encoder(bottleneck):
                if bottleneck:
                    return RelativeRepresentation(rep_dim=rep_dim, hidden_dims=(rep_dim,), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=True)
                else:
                    return RelativeRepresentation(rep_dim=value_hidden_dims[-1], hidden_dims=(value_hidden_dims[-1],), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=False)

            value_state_encoder = make_encoder(bottleneck=False)
            value_goal_encoder = make_encoder(bottleneck=use_waypoints)
            policy_state_encoder = make_encoder(bottleneck=False)
            policy_goal_encoder = make_encoder(bottleneck=False)
            high_policy_state_encoder = make_encoder(bottleneck=False)
            high_policy_goal_encoder = make_encoder(bottleneck=False)
        else:
            def make_encoder(bottleneck):
                if bottleneck:
                    return RelativeRepresentation(rep_dim=rep_dim, hidden_dims=(*value_hidden_dims, rep_dim), layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=True)
                else:
                    return RelativeRepresentation(rep_dim=value_hidden_dims[-1], hidden_dims=(*value_hidden_dims, value_hidden_dims[-1]), layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=False)

            if use_rep:
                value_goal_encoder = make_encoder(bottleneck=True)

        value_def = MonolithicVF(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, rep_dim=rep_dim)

        if discrete:
            action_dim = actions[0] + 1
            actor_def = DiscretePolicy(actor_hidden_dims, action_dim=action_dim)
        else:
            action_dim = actions.shape[-1]
            actor_def = Policy(actor_hidden_dims, action_dim=action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        high_action_dim = observations.shape[-1] if not use_rep else rep_dim
        high_actor_def = Policy(actor_hidden_dims, action_dim=high_action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        network_def = HierarchicalActorCritic(
            encoders={
                'value_state': value_state_encoder,
                'value_goal': value_goal_encoder,
                'policy_state': policy_state_encoder,
                'policy_goal': policy_goal_encoder,
                'high_policy_state': high_policy_state_encoder,
                'high_policy_goal': high_policy_goal_encoder,
            },
            networks={
                'value': value_def,
                'target_value': copy.deepcopy(value_def),
                'actor': actor_def,
                'high_actor': high_actor_def,
            },
            use_waypoints=use_waypoints,
        )
        network_tx = optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr))
        network_params = network_def.init(value_key, observations, observations)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = unfreeze(network.params)
        params['networks_target_value'] = params['networks_value']
        network = network.replace(params=freeze(params))

        config = flax.core.FrozenDict(dict(
            discount=discount, temperature=temperature, high_temperature=high_temperature,
            target_update_rate=tau, pretrain_expectile=pretrain_expectile, way_steps=way_steps, rep_dim=rep_dim,
            policy_train_rep=policy_train_rep,
            use_rep=use_rep, use_waypoints=use_waypoints, discrete=discrete,
        ))

        return JointTrainAgent(rng, network=network, critic=None, value=None, target_value=None, actor=None, config=config)


def get_default_config():
    config = ml_collections.ConfigDict({
        'lr': 3e-4,
        'actor_hidden_dims': (256, 256),
        'value_hidden_dims': (256, 256),
        'discount': 0.99,
        'temperature': 1.0,
        'high_temperature': 1.0,
        'tau': 0.005,
        'pretrain_expectile': 0.7,
        'use_waypoints': 0,
        'use_rep': 0,
        'rep_dim': 64,
        'policy_train_rep': 0,
        'discrete': 1,
    })

    return config


def save_checkpoint(path, agent, step):
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / f"hiql_{step}.msgpack", "wb") as f:
        f.write(flax.serialization.to_bytes(agent))


def train(config):
    seed = int(_get(config, "training.seed", 0))
    random.seed(seed)
    np.random.seed(seed)

    pretrain_dataset = HiQLDataset(
        dataset_dir=_get(config, "dataset.path", "offline_trajectories"),
        max_trajectories=_get(config, "dataset.max_trajectories", None),
        way_steps=int(_get(config, "model.way_steps", 0)),
        p_randomgoal=float(_get(config, "model.p_randomgoal", 0.3)),
        p_trajgoal=float(_get(config, "model.p_trajgoal", 0.5)),
        p_currgoal=float(_get(config, "model.p_currgoal", 0.2)),
        geom_sample=int(_get(config, "model.geom_sample", 0)),
        discount=float(_get(config, "model.discount", 0.99)),
        high_p_randomgoal=float(_get(config, "model.high_p_randomgoal", 0.0)),
        trajectories_per_batch=_get(
            config, "dataset.trajectories_per_batch", None
        ),
    )

    print("\ndataset size:", len(pretrain_dataset))

    batch_size = int(_get(config, "training.batch_size", 64))
    max_steps = int(_get(config, "training.max_steps", 10000))
    log_interval = int(_get(config, "training.log_interval", 100))
    save_interval = int(_get(config, "training.save_interval", 1000))
    checkpoint_dir = _get(config, "training.checkpoint_dir", "checkpoints/hiql")
    train_log_path = _get(
        config, "training.log_path", "hiql_train_log.csv"
    )
    train_logger = CsvLogger(train_log_path)


    one_sampling_start = time.time()
    obs_dim = pretrain_dataset.sample(1)["observations"].shape[-1]
    one_sampling_end = time.time()
    print(f"One batch sampling time: {one_sampling_end - one_sampling_start:.4f} seconds")

    action_dim = int(_get(config, "model.action_dim", 7))
    discrete = int(_get(config, "model.discrete", 1))

    dummy_actions = jnp.array([action_dim - 1], dtype=jnp.int32) if discrete else jnp.zeros((1, action_dim), dtype=jnp.float32)
    agent = create_learner(
        seed=seed,
        observations=jnp.zeros((1, obs_dim), dtype=jnp.float32),
        actions=dummy_actions,
        lr=float(_get(config, "model.lr", 3e-4)),
        actor_hidden_dims=tuple(_get(config, "model.actor_hidden_dims", (256, 256))),
        value_hidden_dims=tuple(_get(config, "model.value_hidden_dims", (256, 256))),
        discount=float(_get(config, "model.discount", 0.99)),
        tau=float(_get(config, "model.tau", 0.005)),
        temperature=float(_get(config, "model.temperature", 1.0)),
        high_temperature=float(_get(config, "model.high_temperature", 1.0)),
        pretrain_expectile=float(_get(config, "model.pretrain_expectile", 0.7)),
        way_steps=int(_get(config, "model.way_steps", 0)),
        rep_dim=int(_get(config, "model.rep_dim", 64)),
        use_rep=int(_get(config, "model.use_rep", 0)),
        policy_train_rep=float(_get(config, "model.policy_train_rep", 0)),
        visual=int(_get(config, "model.visual", 0)),
        encoder=_get(config, "model.encoder", "impala"),
        discrete=discrete,
        use_layer_norm=int(_get(config, "model.use_layer_norm", 0)),
        rep_type=_get(config, "model.rep_type", "state"),
        use_waypoints=int(_get(config, "model.use_waypoints", 0)),
    )

    print("\n Agent initialized. Starting pretraining...\n")
    
    for step in tqdm(range(1, max_steps + 1), 
                     desc="HIQL Training", 
                     unit="step", 
                     dynamic_ncols=True):
        
        batch_sampling_start = time.time()
        batch = pretrain_dataset.sample(batch_size)
        batch_sampling_end = time.time()

        if step == 1:
            print(f"Batch sampling time: {batch_sampling_end - batch_sampling_start:.4f} seconds")
        
        pre_train_s = time.time()
        agent, info = agent.pretrain_update(
            batch,
            seed=seed,
            high_actor_update=bool(agent.config['use_waypoints']),
        )
        pre_train_t = time.time()

        if step == 1:
            pre_train_time = pre_train_t - pre_train_s
            print(f"agent 1 step update time : {pre_train_time:.4f}s", pre_train_time)

        if step % log_interval == 0 or step == 1:
            info = tree_map(lambda x: float(np.asarray(x)) if isinstance(x, (np.ndarray, jnp.ndarray)) else x, info)
            info_str = " ".join([f"{k}={v:.6f}" for k, v in info.items()])
            train_logger.log(dict(info), step)
            # policy_rep_fn = agent.get_policy_rep
            # base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])
            print(f"step={step} {info_str}")
            # progress.set_postfix_str(info_str)

        if step % save_interval == 0 or step == max_steps:
            save_checkpoint(checkpoint_dir, agent, step)
            print(f"saved checkpoint step={step} to {checkpoint_dir}")

    train_logger.close()
