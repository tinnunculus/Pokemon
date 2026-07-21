import copy
import random
import time
from pathlib import Path
from tqdm.auto import tqdm
import time

from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update
from jaxrl_m.networks import Policy, Critic, ensemblize, DiscretePolicy

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


class HiQLTransitionDataset:
    def __init__(self,
                 dataset_dir,
                 max_trajectories=None,
                 p_randomgoal=0.3,
                 p_trajgoal=0.5,
                 p_currgoal=0.2,
                 geom_sample=0,
                 discount=0.99,
                 way_steps=0,
                 high_p_randomgoal=0.0):
        self.dataset_dir = Path(dataset_dir)
        self.files = sorted(self.dataset_dir.glob("*.npz"))
        if max_trajectories is not None:
            self.files = self.files[: int(max_trajectories)]
        if not self.files:
            raise FileNotFoundError(f"No .npz trajectories found in {self.dataset_dir}")

        self.lengths = []
        for path in self.files:
            with np.load(path, mmap_mode='r') as data:
                self.lengths.append(int(data["steps"]))
        self.lengths = np.asarray(self.lengths, dtype=np.int64)
        self.sample_probs = self.lengths / self.lengths.sum()

        self._data_cache = {}

        self.p_randomgoal = float(p_randomgoal)
        self.p_trajgoal = float(p_trajgoal)
        self.p_currgoal = float(p_currgoal)
        self.geom_sample = int(geom_sample)
        self.discount = float(discount)
        self.way_steps = int(way_steps)
        self.high_p_randomgoal = float(high_p_randomgoal)

    def __len__(self):
        return int(self.lengths.sum())

    def _select_goal_index(self, data, current_index):
        goal_locs = np.flatnonzero(data["obs_badges"][:, -1] == 1)
        if len(goal_locs) == 0:
            return int(data["steps"] - 1)

        if current_index <= goal_locs[0]:
            return int(goal_locs[0])
        if current_index >= goal_locs[-1]:
            return int(goal_locs[-1])

        return int(goal_locs[np.searchsorted(goal_locs, current_index)])

    def _load_cached_data(self, path):
        path_key = str(path)
        if path_key not in self._data_cache:
            load_start = time.time()
            self._data_cache[path_key] = np.load(path, mmap_mode='r')
            load_end = time.time()
            # print(f"Loaded {path} in {load_end - load_start:.4f} seconds")
        return self._data_cache[path_key]

    def _sample_goal_index(self, data, current_index):
        traj_len = int(data["steps"])
        goal_locs = np.flatnonzero(data["obs_badges"][:, -1] == 1)
        if len(goal_locs) == 0:
            return int(traj_len - 1)

        final_state_index = int(goal_locs[-1])
        random_goal_index = int(np.random.randint(traj_len))

        if self.p_currgoal > 0 and np.random.rand() < self.p_currgoal:
            return int(current_index)

        if self.p_trajgoal > 0 and np.random.rand() < self.p_trajgoal / (1.0 - self.p_currgoal):
            if self.geom_sample:
                us = np.random.rand()
                return int(np.minimum(current_index + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_index))
            distance = np.random.rand()
            return int(np.round((np.minimum(current_index + 1, final_state_index) * distance + final_state_index * (1 - distance))).astype(int))

        return int(random_goal_index)

    def _sample_goal_index_batch(self, data, current_indices):
        traj_len = int(data["steps"])
        goal_locs = np.flatnonzero(data["obs_badges"][:, -1] == 1)
        if len(goal_locs) == 0:
            return np.full(current_indices.shape, traj_len - 1, dtype=np.int64)

        final_state_index = int(goal_locs[-1])
        random_goal_index = np.random.randint(traj_len, size=current_indices.shape)

        goal_indices = np.empty_like(current_indices, dtype=np.int64)
        curr_mask = (self.p_currgoal > 0) & (np.random.rand(current_indices.shape[0]) < self.p_currgoal)
        traj_mask = (~curr_mask) & (self.p_trajgoal > 0) & (np.random.rand(current_indices.shape[0]) < self.p_trajgoal / (1.0 - self.p_currgoal))

        goal_indices[curr_mask] = current_indices[curr_mask]

        if self.geom_sample:
            us = np.random.rand(current_indices.shape[0])
            new_goal_idx = np.minimum(
                current_indices[traj_mask] + np.ceil(np.log(1 - us[traj_mask]) / np.log(self.discount)).astype(int),
                final_state_index,
            )
            goal_indices[traj_mask] = new_goal_idx
        else:
            distance = np.random.rand(current_indices.shape[0])
            new_goal_idx = np.round((
                np.minimum(current_indices[traj_mask] + 1, final_state_index) * distance[traj_mask]
                + final_state_index * (1 - distance[traj_mask])
            )).astype(int)
            goal_indices[traj_mask] = new_goal_idx

        goal_indices[~curr_mask & ~traj_mask] = random_goal_index[~curr_mask & ~traj_mask]
        return goal_indices

    def _sample_one(self, data=None):
        if data is None:
            traj_idx = int(np.random.choice(len(self.files), p=self.sample_probs))
            path = self.files[traj_idx]
            data = self._load_cached_data(path)

        traj_len = int(data["steps"])
        index = int(np.random.randint(traj_len))
        goal_index = self._sample_goal_index(data, index)
        final_state_index = int(np.flatnonzero(data["obs_badges"][:, -1] == 1)[-1]) if np.any(data["obs_badges"][:, -1] == 1) else int(traj_len - 1)

        obs = {key: data[f"obs_{key}"][index] for key in OBS_KEYS}
        next_obs = {key: data[f"next_obs_{key}"][index] for key in OBS_KEYS}
        goal_obs = {key: data[f"obs_{key}"][goal_index] for key in OBS_KEYS}

        observations = flatten_observation(obs)
        next_observations = flatten_observation(next_obs)
        goals = flatten_observation(goal_obs)

        success = np.float32(1.0 if index == goal_index else 0.0)
        low_goal_index = int(np.minimum(index + self.way_steps, final_state_index))
        low_goal_obs = {key: data[f"obs_{key}"][low_goal_index] for key in OBS_KEYS}
        low_goals = flatten_observation(low_goal_obs)

        high_goal_index = goal_index
        high_target_index = int(np.minimum(index + self.way_steps, final_state_index))
        if self.high_p_randomgoal > 0 and np.random.rand() < self.high_p_randomgoal:
            high_goal_index = int(np.random.randint(traj_len))
            high_target_index = int(np.minimum(index + self.way_steps, final_state_index))

        high_goal_obs = {key: data[f"obs_{key}"][high_goal_index] for key in OBS_KEYS}
        high_target_obs = {key: data[f"obs_{key}"][high_target_index] for key in OBS_KEYS}
        high_goals = flatten_observation(high_goal_obs)
        high_targets = flatten_observation(high_target_obs)

        return {
            "observations": observations,
            "next_observations": next_observations,
            "actions": np.int64(data["actions"][index]),
            "rewards": success,
            "masks": np.float32(1.0 - success),
            "goals": goals,
            "low_goals": low_goals,
            "high_goals": high_goals,
            "high_targets": high_targets,
        }

    def __getitem__(self, _):
        return self._sample_one()

    def sample_batch(self, batch_size):
        traj_idx = int(np.random.choice(len(self.files), p=self.sample_probs))
        path = self.files[traj_idx]
        data = self._load_cached_data(path)

        traj_len = int(data["steps"])
        indices = np.random.randint(traj_len, size=batch_size)
        goal_indices = self._sample_goal_index_batch(data, indices)
        final_state_index = int(np.flatnonzero(data["obs_badges"][:, -1] == 1)[-1]) if np.any(data["obs_badges"][:, -1] == 1) else int(traj_len - 1)

        obs = {key: data[f"obs_{key}"][indices] for key in OBS_KEYS}
        next_obs = {key: data[f"next_obs_{key}"][indices] for key in OBS_KEYS}
        goal_obs = {key: data[f"obs_{key}"][goal_indices] for key in OBS_KEYS}

        observations = flatten_observation_batch(obs)
        next_observations = flatten_observation_batch(next_obs)
        goals = flatten_observation_batch(goal_obs)

        success = (indices == goal_indices).astype(np.float32)
        low_goal_indices = np.minimum(indices + self.way_steps, final_state_index)
        low_goal_obs = {key: data[f"obs_{key}"][low_goal_indices] for key in OBS_KEYS}
        low_goals = flatten_observation_batch(low_goal_obs)

        high_goal_indices = goal_indices
        high_target_indices = np.minimum(indices + self.way_steps, final_state_index)
        if self.high_p_randomgoal > 0:
            pick_random = np.random.rand(batch_size) < self.high_p_randomgoal
            high_goal_indices = np.where(pick_random, np.random.randint(traj_len, size=batch_size), high_goal_indices)

        high_goal_obs = {key: data[f"obs_{key}"][high_goal_indices] for key in OBS_KEYS}
        high_target_obs = {key: data[f"obs_{key}"][high_target_indices] for key in OBS_KEYS}
        high_goals = flatten_observation_batch(high_goal_obs)
        high_targets = flatten_observation_batch(high_target_obs)

        batch = {
            "observations": observations,
            "next_observations": next_observations,
            "actions": np.asarray(data["actions"][indices], dtype=np.int64),
            "rewards": success,
            "masks": np.float32(1.0 - success),
            "goals": goals,
            "low_goals": low_goals,
            "high_goals": high_goals,
            "high_targets": high_targets,
        }
        return {
            key: jnp.asarray(value)
            for key, value in batch.items()
        }


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

    def act(agent, obs, device="cpu", deterministic=True, temperature: float = 1.0):
        if isinstance(obs, dict):
            observations = flatten_observation(obs)
        else:
            observations = np.asarray(obs, dtype=np.float32)

        if observations.ndim == 1:
            observations = observations[None, :]

        goals = observations
        discrete = int(agent.config.get('discrete', 0))

        if deterministic:
            dist = agent.network(observations, goals, low_dim_goals=False, temperature=temperature, method='actor')
            action = dist.mode()
        else:
            seed = jax.random.PRNGKey(0)
            action = agent.sample_actions(observations, goals, seed=seed, temperature=temperature, discrete=discrete)

        if hasattr(action, 'shape') and len(action.shape) > 1:
            action = action[0]

        if hasattr(action, 'item'):
            try:
                return action.item()
            except Exception:
                pass

        return np.asarray(action)

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
    with open(checkpoint_dir / f"checkpoint_{step:09d}.msgpack", "wb") as f:
        f.write(flax.serialization.to_bytes(agent))


def train(config):
    seed = int(_get(config, "training.seed", 0))
    random.seed(seed)
    np.random.seed(seed)

    dataset = HiQLTransitionDataset(
        dataset_dir=_get(config, "dataset.path", "offline_trajectories"),
        max_trajectories=_get(config, "dataset.max_trajectories", None),
        way_steps=int(_get(config, "model.way_steps", 0)),
        p_randomgoal=float(_get(config, "model.p_randomgoal", 0.3)),
        p_trajgoal=float(_get(config, "model.p_trajgoal", 0.5)),
        p_currgoal=float(_get(config, "model.p_currgoal", 0.2)),
        geom_sample=int(_get(config, "model.geom_sample", 0)),
        discount=float(_get(config, "model.discount", 0.99)),
        high_p_randomgoal=float(_get(config, "model.high_p_randomgoal", 0.0)),
    )

    print("\ndataset size:", len(dataset))

    batch_size = int(_get(config, "training.batch_size", 64))
    max_steps = int(_get(config, "training.max_steps", 10000))
    log_interval = int(_get(config, "training.log_interval", 100))
    save_interval = int(_get(config, "training.save_interval", 1000))
    checkpoint_dir = _get(config, "training.checkpoint_dir", "checkpoints/hiql")


    one_sampling_start = time.time()
    obs_dim = dataset.sample_batch(1)["observations"].shape[-1]
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

    print("\n Environment and agent initialized. Starting pretraining...\n")

    for step in tqdm(range(1, max_steps + 1), desc="HIQL Training", unit="step"):
        # print("step:", step)
        batch_sampling_start = time.time()
        batch = dataset.sample_batch(batch_size)
        batch_sampling_end = time.time()
        # print(f"Batch sampling time: {batch_sampling_end - batch_sampling_start:.4f} seconds")
        
        """
        
        if step < 5: 
            batch_np = {
                k: np.asarray(v)
                for k, v in batch.items()
            }

            reward_sum = batch_np["rewards"].sum()
            mask_sum = batch_np["masks"].sum()
            goal_match = (batch_np["rewards"] == 1).mean()
            
            print(
                f"[dataset-check] reward_mean={batch_np['rewards'].mean():.4f} "
                f"mask_mean={batch_np['masks'].mean():.4f} "
                f"reward+mask_sum={reward_sum + mask_sum:.4f} "   # higher than 1.0
                f"goal_match_rate={goal_match:.4f}"    # between 0~1
                f"final badge observation: {batch_np['goals'][0][-10:]}"
            )
        """

        agent, info = agent.pretrain_update(batch, seed=seed + step, value_update=True, actor_update=True, high_actor_update=False)

        if step % log_interval == 0 or step == 1:
            info = tree_map(lambda x: float(np.asarray(x)) if isinstance(x, (np.ndarray, jnp.ndarray)) else x, info)
            info_str = " ".join([f"{k}={v:.6f}" for k, v in info.items()])
            print(f"step={step} {info_str}")
            # progress.set_postfix_str(info_str)

        if step % save_interval == 0 or step == max_steps:
            save_checkpoint(checkpoint_dir, agent, step)
            print(f"saved checkpoint step={step} to {checkpoint_dir}")

