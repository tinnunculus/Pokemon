import os
from os.path import exists
from pathlib import Path
import uuid
import time
import glob
import random
import tempfile
import numpy as np
import torch
from env.red_gym_env_v2 import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

START_SEED = 0
NUM_TRAJECTORIES = 300
MAX_TRAJECTORY_STEPS = 50_000
DATASET_DIR = Path("offline_trajectories")
DETERMINISTIC_ACTIONS = False

def fix_seed(seed):
    # Python, NumPy, PyTorch, Stable-Baselines3에서 사용하는 난수를 같은 값으로 고정한다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_random_seed(seed)

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # Stable-Baselines3의 VecEnv 생성 방식에 맞춰 환경 생성을 지연시킨다.
        env = RedGymEnv(env_conf)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed + rank)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed + rank)
        return env
    # 여러 환경을 띄울 때 재현 가능한 난수 흐름을 만들기 위한 기본 시드 설정.
    set_random_seed(seed + rank)
    return _init

def get_most_recent_zip_with_age(folder_path):
    # 지정한 폴더 안의 학습 체크포인트(.zip) 목록을 찾는다.
    zip_files = glob.glob(os.path.join(folder_path, "*.zip"))
    
    if not zip_files:
        return None, None  # 체크포인트가 없으면 호출부에서 처리할 수 있게 None을 반환한다.
    
    # 수정 시간이 가장 최근인 체크포인트를 선택한다.
    most_recent_zip = max(zip_files, key=os.path.getmtime)
    
    # 선택한 체크포인트가 몇 시간 전에 저장되었는지 계산한다.
    current_time = time.time()
    modification_time = os.path.getmtime(most_recent_zip)
    age_in_hours = (current_time - modification_time) / 3600  # 초 단위를 시간 단위로 변환한다.
    
    return most_recent_zip, age_in_hours

def append_obs(storage, obs):
    for key, value in obs.items():
        storage.setdefault(key, []).append(np.array(value, copy=True))

def stack_obs(storage):
    return {key: np.stack(values, axis=0) for key, values in storage.items()}

def save_trajectory(output_dir, seed, checkpoint, observations, next_observations,
                    actions, rewards, terminated, truncated, final_obs):
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_arrays = stack_obs(observations)
    next_obs_arrays = stack_obs(next_observations)

    save_data = {
        "seed": np.array(seed, dtype=np.int64),
        "steps": np.array(len(actions), dtype=np.int64),
        "checkpoint": np.array(checkpoint),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
        "dones": np.asarray(terminated, dtype=np.bool_) | np.asarray(truncated, dtype=np.bool_),
        "final_badges": np.asarray(final_obs["badges"]),
    }
    for key, value in obs_arrays.items():
        save_data[f"obs_{key}"] = value
    for key, value in next_obs_arrays.items():
        save_data[f"next_obs_{key}"] = value

    file_path = output_dir / f"trajectory_seed_{seed:03d}.npz"
    np.savez_compressed(file_path, **save_data)
    return file_path

def collect_trajectory(model, env_config, seed, checkpoint):
    fix_seed(seed)
    seed_env_config = dict(env_config)
    # RedGymEnv requires a session_path and writes debug frames there.
    # Use a temporary directory so no persistent session folder is created.
    with tempfile.TemporaryDirectory(prefix=f"pokemon_offline_seed_{seed:03d}_") as session_dir:
        seed_env_config["session_path"] = Path(session_dir)
        env = make_env(0, seed_env_config, seed=seed)()
        model.set_random_seed(seed)

        observations = {}
        next_observations = {}
        actions = []
        rewards = []
        terminated_flags = []
        truncated_flags = []

        obs, info = env.reset(seed=seed)
        steps = 0
        try:
            while True:
                if steps >= MAX_TRAJECTORY_STEPS:
                    break
                if obs["badges"][-1] != 0:
                    break

                action, _states = model.predict(obs, deterministic=DETERMINISTIC_ACTIONS)
                next_obs, reward, terminated, truncated, info = env.step(action)

                append_obs(observations, obs)
                append_obs(next_observations, next_obs)
                actions.append(int(np.asarray(action).item()))
                rewards.append(float(reward))
                terminated_flags.append(bool(terminated))
                truncated_flags.append(bool(truncated))

                obs = next_obs
                steps += 1
                if terminated or truncated:
                    break
        finally:
            env.close()

    file_path = save_trajectory(
        DATASET_DIR,
        seed,
        checkpoint,
        observations,
        next_observations,
        actions,
        rewards,
        terminated_flags,
        truncated_flags,
        obs,
    )
    print(f"saved {file_path} | steps={steps} | badges={obs['badges']}")
    return file_path, steps

if __name__ == '__main__':
    fix_seed(START_SEED)

    ep_length = 2**23

    # RedGymEnv에 전달할 실행 환경 설정.
    # headless=True라서 창을 띄우지 않고, init.state와 PokemonRed.gb는 상위 폴더에서 읽는다.
    env_config = {
                'headless': True, 'save_final_state': False, 'early_stop': False,
                'action_freq': 24, 'init_state': 'env/init.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True,
                'gb_path': 'red_rom/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': False
            }

    # runs 폴더에서 가장 최근 PPO 체크포인트를 자동으로 불러온다.
    most_recent_checkpoint, time_since = get_most_recent_zip_with_age("online_weights")
    if most_recent_checkpoint is not None:
        file_name = most_recent_checkpoint
        print(f"using checkpoint: {file_name}, which is {time_since} hours old")
    else:
        raise FileNotFoundError("online_weights 폴더에서 PPO 체크포인트(.zip)를 찾지 못했습니다.")
    
    # 필요하면 자동 선택 대신 특정 체크포인트를 직접 지정할 수 있다.
    #file_name = "runs/poke_41943040_steps.zip"
    print('\nloading checkpoint')
    # 저장 당시의 스케줄 객체가 없어도 로드되도록 lr_schedule과 clip_range를 상수로 대체한다.
    model = PPO.load(file_name, custom_objects={'lr_schedule': 0, 'clip_range': 0})

    total_steps = 0
    for seed in range(START_SEED, START_SEED + NUM_TRAJECTORIES):
        _, steps = collect_trajectory(model, env_config, seed, file_name)
        total_steps += steps

    print(f"saved {NUM_TRAJECTORIES} trajectories to {DATASET_DIR}")
    print(f"total_steps: {total_steps}")
