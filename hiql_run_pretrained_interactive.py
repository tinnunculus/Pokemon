import argparse
import pathlib
import uuid
from pathlib import Path

import flax
import jax.numpy as jnp
import numpy as np
import torch

from env.red_gym_env_v2 import RedGymEnv
from models.hiql import create_learner, flatten_observation


SCRIPT_DIR = Path(__file__).resolve().parent
print("script dir:", SCRIPT_DIR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        return env

    return _init


def resolve_checkpoint(path=None):
    if path is None:
        checkpoint_dir = SCRIPT_DIR / "checkpoints" / "hiql"
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.msgpack"))
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
        help="Path to a HIQL .msgpack checkpoint. Defaults to the latest checkpoint in checkpoints/hiql.",
    )
    return parser.parse_args()


def load_hiql_model(checkpoint_path, env):
    obs, _ = env.reset()
    obs_dim = flatten_observation(obs).shape[-1]

    action_dim = 7
    discrete = 1
    dummy_actions = jnp.array([action_dim - 1], dtype=jnp.int32) if discrete else jnp.zeros((1, action_dim), dtype=jnp.float32)

    agent = create_learner(
        seed=0,
        observations=jnp.zeros((1, obs_dim), dtype=jnp.float32),
        actions=dummy_actions,
        lr=3e-4,
        actor_hidden_dims=(256, 256),
        value_hidden_dims=(256, 256),
        discount=0.99,
        tau=0.005,
        temperature=1.0,
        high_temperature=1.0,
        pretrain_expectile=0.7,
        way_steps=0,
        rep_dim=64,
        use_rep=0,
        policy_train_rep=0,
        visual=0,
        encoder="impala",
        discrete=discrete,
        use_layer_norm=0,
        rep_type="state",
        use_waypoints=0,
    )

    with open(checkpoint_path, "rb") as f:
        checkpoint_bytes = f.read()

    agent = flax.serialization.from_bytes(agent, checkpoint_bytes)
    print(f"Loaded HIQL checkpoint from: {checkpoint_path}")
    return agent


if __name__ == "__main__":
    args = parse_args()
    print(f"Using device: {DEVICE}")

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
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    model = load_hiql_model(checkpoint_path, env)

    obs, info = env.reset()

    while True:
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except Exception:
            agent_enabled = False

        if agent_enabled:
            action = model.act(obs, device=DEVICE, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            print(f"step={env.step_count}, action={action}")
            obs, rewards, terminated, truncated, info = env.step(action)
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
