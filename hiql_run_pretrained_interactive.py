import argparse
import pathlib
import uuid
from pathlib import Path

import flax
import jax.numpy as jnp
from jaxrl_m.evaluation import supply_rng, evaluate_with_trajectories, EpisodeMonitor
import numpy as np
import torch

from env.red_gym_env_v2 import RedGymEnv
from models.hiql import create_learner, flatten_observation
from main import load_simple_yaml


SCRIPT_DIR = Path(__file__).resolve().parent
print("script dir:", SCRIPT_DIR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        return env

    return _init


def resolve_checkpoint(path=None, checkpoint_dir="checkpoints/hiql_waypoints"):
    if path is None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = SCRIPT_DIR / checkpoint_dir
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("hiql_*.msgpack"))
            if checkpoints:
                return checkpoints[-1]
        raise FileNotFoundError(f"No HIQL checkpoint found in {checkpoint_dir}")

    checkpoint = Path(path)
    if not checkpoint.is_absolute():
        checkpoint = SCRIPT_DIR / checkpoint
    if checkpoint.exists():
        return checkpoint
    raise FileNotFoundError(f"HIQL checkpoint not found: {checkpoint}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained HIQL policy in Pokemon Red.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a HIQL .msgpack checkpoint. Defaults to the latest checkpoint in training.checkpoint_dir.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hiql.yaml",
        help="HIQL training config. Its model architecture must match the checkpoint.",
    )
    parser.add_argument(
        "--goal-trajectory",
        type=str,
        default=None,
        help="Successful .npz trajectory used to obtain the final badge goal. Defaults to the first successful dataset file.",
    )
    parser.add_argument(
        "--waypoint-replan-steps",
        type=int,
        default=None,
        help="Primitive-decision interval for replanning. Defaults to model.way_steps.",
    )
    return parser.parse_args()


def load_goal_observation(dataset_dir, trajectory_path=None):
    """Return a real successful observation, not a synthetic badges-only goal."""
    if trajectory_path is not None:
        paths = [Path(trajectory_path)]
        if not paths[0].is_absolute():
            paths[0] = SCRIPT_DIR / paths[0]
    else:
        paths = sorted(dataset_dir.glob("*.npz"))

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Goal trajectory not found: {path}")
        with np.load(path, mmap_mode="r") as data:
            success_indices = np.flatnonzero(
                (data["obs_badges"][:, -1] == 1) | (data["next_obs_badges"][:, -1] == 1)
            )
            if len(success_indices):
                index = int(success_indices[0])
                goal_prefix = "next_obs" if data["next_obs_badges"][index, -1] == 1 else "obs"
                goal_obs = {
                    key: np.array(data[f"{goal_prefix}_{key}"][index], copy=True)
                    for key in ("screens", "health", "level", "badges", "events", "map", "recent_actions")
                }
                print(f"Using badge-success goal from {path.name}, trajectory step {index}")
                return goal_obs

    raise ValueError(
        f"No trajectory with obs_badges[:, -1] == 1 or next_obs_badges[:, -1] == 1 was found in {dataset_dir}. "
        "A hierarchical policy needs at least one successful demonstration."
    )


def load_hiql_model(checkpoint_path, env, model_config):
    obs, _ = env.reset()
    obs_dim = flatten_observation(obs).shape[-1]

    action_dim = int(model_config.get("action_dim", 7))
    discrete = int(model_config.get("discrete", 1))
    dummy_actions = jnp.array([action_dim - 1], dtype=jnp.int32) if discrete else jnp.zeros((1, action_dim), dtype=jnp.float32)

    agent = create_learner(
        seed=0,
        observations=jnp.zeros((1, obs_dim), dtype=jnp.float32),
        actions=dummy_actions,
        lr=float(model_config.get("lr", 3e-4)),
        actor_hidden_dims=tuple(model_config.get("actor_hidden_dims", (256, 256))),
        value_hidden_dims=tuple(model_config.get("value_hidden_dims", (256, 256))),
        discount=float(model_config.get("discount", 0.99)),
        tau=float(model_config.get("tau", 0.005)),
        temperature=float(model_config.get("temperature", 1.0)),
        high_temperature=float(model_config.get("high_temperature", 1.0)),
        pretrain_expectile=float(model_config.get("pretrain_expectile", 0.7)),
        way_steps=int(model_config.get("way_steps", 0)),
        rep_dim=int(model_config.get("rep_dim", 64)),
        use_rep=int(model_config.get("use_rep", 0)),
        policy_train_rep=float(model_config.get("policy_train_rep", 0)),
        visual=int(model_config.get("visual", 0)),
        encoder=model_config.get("encoder", "impala"),
        discrete=discrete,
        use_layer_norm=int(model_config.get("use_layer_norm", 0)),
        rep_type=model_config.get("rep_type", "state"),
        use_waypoints=int(model_config.get("use_waypoints", 0)),
    )

    with open(checkpoint_path, "rb") as f:
        checkpoint_bytes = f.read()

    agent = flax.serialization.from_bytes(agent, checkpoint_bytes)
    print(f"Loaded HIQL checkpoint from: {checkpoint_path}")
    return agent


if __name__ == "__main__":
    args = parse_args()
    print(f"Using device: {DEVICE}")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = SCRIPT_DIR / config_path
    config = load_simple_yaml(config_path)
    model_config = config.get("model", {})
    dataset_config = config.get("dataset", {})
    training_config = config.get("training", {})

    sess_path = SCRIPT_DIR / f"session_{str(uuid.uuid4())[:8]}"
    ep_length = 2**23

    env_config = {
        "headless": False,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "init.state",
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": "red_rom/PokemonRed.gb",
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "extra_buttons": False,
    }

    env = make_env(0, env_config)()
    checkpoint_path = resolve_checkpoint(
        args.checkpoint,
        training_config.get("checkpoint_dir", "checkpoints/hiql_waypoints"),
    )
    model = load_hiql_model(checkpoint_path, env, model_config)
    dataset_dir = Path(dataset_config.get("path", "offline_trajectories"))
    if not dataset_dir.is_absolute():
        dataset_dir = SCRIPT_DIR / dataset_dir
    goal_obs = load_goal_observation(dataset_dir, args.goal_trajectory)
    replan_steps = args.waypoint_replan_steps or int(model_config.get("way_steps", 1))
    if replan_steps < 1:
        raise ValueError("waypoint replan interval must be at least 1")

    obs, info = env.reset()
    waypoint = None
    next_replan_step = 0

    while True:
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except Exception:
            agent_enabled = False

        if agent_enabled:
            if model.config["use_waypoints"] and env.step_count >= next_replan_step:
                waypoint = model.get_waypoint(obs, goal_obs, deterministic=True)
                next_replan_step = env.step_count + replan_steps
                print(f"step={env.step_count}, replanned waypoint (next replan: {next_replan_step})")
            action = model.act(obs, goal_obs, waypoint=waypoint, device=DEVICE, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            print(f"step={env.step_count}, action={action}")
            obs, rewards, terminated, truncated, info = env.step(action)
            print("rewards:", rewards)
        else:
            env.pyboy.tick(1, True)
            obs = env._get_obs()
            truncated = env.step_count >= env.max_steps - 1

        env.render()

        if isinstance(obs, dict) and "badges" in obs and len(obs["badges"]) > 0 and obs["badges"][-1] == 1:
            print("got a badge!!!")
            env.close()
            break

        if truncated:
            break

    env.close()
