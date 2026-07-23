import argparse
import pathlib
import random
import uuid
from pathlib import Path
from pyboy.utils import WindowEvent


import numpy as np
import torch

from env.red_gym_env_v2 import RedGymEnv
from models.cql import CQLDQN

from stable_baselines3.common.utils import set_random_seed



SCRIPT_DIR = Path(__file__).resolve().parent
print("script dir:", SCRIPT_DIR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        #env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init



def resolve_checkpoint(path=None):
    if path is not None:
        checkpoint = Path(path)
        if not checkpoint.is_absolute():
            checkpoint = SCRIPT_DIR / checkpoint
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"CQL checkpoint not found: {checkpoint}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained CQLDQN in Pokemon Red.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a CQL .pt checkpoint. Defaults to cql_finalp.pt, then cql_final.pt.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Using device: {DEVICE}")

    sess_path = SCRIPT_DIR / f"session_{str(uuid.uuid4())[:8]}"
    ep_length = 2**23

    env_config = {
                'headless': False, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': 'init.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': 'red_rom/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': False
            }
    

    env = make_env(0, env_config)() 

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    print(f"\nloading checkpoint: {checkpoint_path}")
    model = CQLDQN.load(checkpoint_path, env=env, device=DEVICE)
    
    obs, info = env.reset()

    while True:
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False

        if agent_enabled:
            action = model.act(obs, device=DEVICE, deterministic=True)
            print(f"step={env.step_count}, action={int(np.asarray(action).item())}")
            obs, rewards, terminated, truncated, info = env.step(action)
            print("rewards", rewards)
        else:
            env.pyboy.tick(1, True)
            obs = env._get_obs()
            truncated = env.step_count >= env.max_steps - 1
        env.render()

        if truncated:
            break
    env.close()

