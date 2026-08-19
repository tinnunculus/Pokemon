"""Run a pretrained HIQL agent interactively in Pokemon Red.

The agent advances in evaluation chunks of 100 agent steps.  Writing a line
starting with ``yes`` to agent_enabled.txt enables the agent; any other value
leaves PyBoy running for manual play without consuming the agent step budget.
"""

import argparse
import copy
import re
import uuid
from functools import partial
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from env.red_gym_env_v2 import RedGymEnv
from jaxrl_m.evaluation import evaluate_with_trajectories, supply_rng
from main import load_simple_yaml
from models.hiql import OBS_KEYS, create_learner, flatten_observation


SCRIPT_DIR = Path(__file__).resolve().parent
MAX_STEPS = 50_000
EVALUATION_INTERVAL = 100


def _get(config, key, default=None):
    current = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def resolve_from_script_dir(path):
    path = Path(path)
    return path if path.is_absolute() else SCRIPT_DIR / path


def _checkpoint_step(path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def resolve_checkpoint(path, config):
    if path is not None:
        checkpoint = resolve_from_script_dir(path)
        if checkpoint.is_file() and checkpoint.suffix == ".msgpack":
            return checkpoint
        raise FileNotFoundError(f"HIQL .msgpack checkpoint not found: {checkpoint}")

    checkpoint_dir = resolve_from_script_dir(
        _get(config, "training.checkpoint_dir", "checkpoints/hiql")
    )
    checkpoints = list(checkpoint_dir.glob("*.msgpack"))
    if not checkpoints:
        # Training runs are often moved to a dated directory after the config
        # was written. Fall back to all HIQL checkpoints in checkpoints/.
        checkpoints = list((SCRIPT_DIR / "checkpoints").glob("**/hiql_*.msgpack"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No HIQL .msgpack checkpoints found in {checkpoint_dir} or "
            f"{SCRIPT_DIR / 'checkpoints'}. Pass one with --checkpoint."
        )
    return max(
        checkpoints,
        key=lambda item: (_checkpoint_step(item), item.stat().st_mtime, str(item)),
    )


def is_agent_enabled(path):
    try:
        with path.open("r", encoding="utf-8") as file:
            return file.readline().lstrip().lower().startswith("yes")
    except (OSError, IndexError):
        return False


def badge_goal(observation):
    """Keep the current state as the goal, except for the final badge bit."""
    goal = copy.deepcopy(observation)
    goal["badges"] = np.array(goal["badges"], copy=True)
    goal["badges"][-1] = 1
    return flatten_observation(goal)


def load_successful_goal(dataset_dir, trajectory_path=None):
    """Load an in-distribution badge goal from a successful trajectory."""
    if trajectory_path is not None:
        paths = [resolve_from_script_dir(trajectory_path)]
    else:
        paths = sorted(dataset_dir.glob("*.npz"))

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Goal trajectory not found: {path}")
        with np.load(path) as trajectory:
            if "final_badges" in trajectory and trajectory["final_badges"][-1] != 1:
                continue
            successful = np.flatnonzero(trajectory["next_obs_badges"][:, -1] == 1)
            if not len(successful):
                continue
            index = int(successful[0])
            goal = {
                key: np.array(trajectory[f"next_obs_{key}"][index], copy=True)
                for key in OBS_KEYS
            }
        print(f"Using badge-success goal: {path} (step {index})")
        return flatten_observation(goal)

    if trajectory_path is not None:
        raise ValueError(f"Goal trajectory does not contain badges[-1] == 1: {paths[0]}")
    return None


def final_badge_reached(observation):
    return bool(np.asarray(observation["badges"])[-1] == 1)


class TrackingEnv:
    """Remember the raw observation hidden by HIQL's flattened trajectory."""

    def __init__(self, env, observation):
        self.env = env
        self.last_observation = observation

    def step(self, action):
        result = self.env.step(action)
        self.last_observation = result[0]
        return result

    def __getattr__(self, name):
        return getattr(self.env, name)


def build_agent(config, observation, checkpoint_path):
    model_config = config.get("model", {})
    discrete = int(model_config.get("discrete", 1))
    action_dim = int(model_config.get("action_dim", 7))
    observations = jnp.zeros(
        (1, flatten_observation(observation).shape[-1]), dtype=jnp.float32
    )
    dummy_actions = (
        jnp.array([action_dim - 1], dtype=jnp.int32)
        if discrete
        else jnp.zeros((1, action_dim), dtype=jnp.float32)
    )

    agent = create_learner(
        seed=int(_get(config, "training.seed", 0)),
        observations=observations,
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
    try:
        return flax.serialization.from_bytes(agent, checkpoint_path.read_bytes())
    except Exception as error:
        raise ValueError(
            "Checkpoint restoration failed. Ensure --config is the same config "
            f"used to train {checkpoint_path}."
        ) from error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a pretrained HIQL agent interactively in Pokemon Red."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="HIQL .msgpack checkpoint (defaults to the latest in the configured directory).",
    )
    parser.add_argument(
        "--config",
        default="configs/hiql.yaml",
        help="HIQL config used to create the checkpoint architecture.",
    )
    parser.add_argument(
        "--agent-enabled-file",
        default="agent_enabled.txt",
        help="File whose first line enables the agent when it starts with 'yes'.",
    )
    parser.add_argument(
        "--goal-trajectory",
        default=None,
        help="Optional successful .npz trajectory to use as the HIQL goal.",
    )
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=0.0,
        help=(
            "Policy sampling temperature. 0 is nearly deterministic; try "
            "0.3-1.0 to sample alternative actions when the agent is stuck."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = resolve_from_script_dir(args.config)
    config = load_simple_yaml(config_path)
    checkpoint_path = resolve_checkpoint(args.checkpoint, config)
    enabled_path = resolve_from_script_dir(args.agent_enabled_file)
    if args.eval_temperature < 0:
        raise ValueError("--eval-temperature must be greater than or equal to 0")

    session_path = SCRIPT_DIR / f"session_{str(uuid.uuid4())[:8]}"
    env_config = {
        "headless": False,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str(SCRIPT_DIR / "init.state"),
        "max_steps": MAX_STEPS,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": session_path,
        "gb_path": str(SCRIPT_DIR / "red_rom" / "PokemonRed.gb"),
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "extra_buttons": False,
    }

    env = RedGymEnv(env_config)
    try:
        observation, _ = env.reset()
        tracking_env = TrackingEnv(env, observation)
        agent = build_agent(config, observation, checkpoint_path)
        dataset_dir = resolve_from_script_dir(
            _get(config, "dataset.path", "offline_trajectories")
        )
        successful_goal = load_successful_goal(dataset_dir, args.goal_trajectory)
        if successful_goal is None:
            print(
                "No successful trajectory was found; using the current observation "
                "with badges[-1] set to 1 as the goal."
            )

        seed = int(_get(config, "training.seed", 0))
        discrete = int(_get(config, "model.discrete", 1))

        # low-level policy: 현재 observation과 waypoint(goal representation) 받아 0~6 중 하나의 action을 선택
        # supply_rng는 호출할 때마다 새로운 JAX random key를 전달한다.
        policy_fn = supply_rng(
            partial(agent.sample_actions, discrete=discrete),
            rng=jax.random.PRNGKey(seed),
        )

        # high-level policy: 현재 observation에서 최종 badge goal로 가기 위한 중간 waypoint를 선택
        high_policy_fn = supply_rng(
            agent.sample_high_actions,
            rng=jax.random.PRNGKey(seed + 1),
        )

        print(f"Loaded HIQL checkpoint: {checkpoint_path}")
        print(f"Agent toggle file: {enabled_path}")
        print(f"Maximum agent steps: {MAX_STEPS}")
        print(f"Evaluation temperature: {args.eval_temperature}")

        while env.step_count < MAX_STEPS:
            observation = tracking_env.last_observation
            if final_badge_reached(observation):
                print(f"Goal reached at step={env.step_count}: badges[-1] == 1")
                break

            if not is_agent_enabled(enabled_path):
                env.pyboy.tick(1, True)
                tracking_env.last_observation = env._get_obs()
                env.render()
                continue

            chunk_steps = min(EVALUATION_INTERVAL, MAX_STEPS - env.step_count)
            start_step = env.step_count

            # evaluate_with_trajectories가 아래 순서를 최대 100회 반복한다.
            #   1. raw Pokemon observation을 HIQL 입력 vector로 flatten
            #   2. high-level policy로 badge goal을 향한 waypoint 선택
            #   3. low-level policy로 실제 PyBoy action 선택
            #   4. env.step(action) 실행 및 transition 기록

            stats, _, _ = evaluate_with_trajectories(
                policy_fn=policy_fn,
                high_policy_fn=high_policy_fn,
                policy_rep_fn=agent.get_policy_rep,
                env=tracking_env,
                env_name="pokemon-red",
                num_episodes=1,
                use_waypoints=bool(agent.config["use_waypoints"]),
                # 0이면 거의 deterministic하게, 값이 커질수록 action과
                # high-level waypoint를 더 stochastic하게 sample한다.
                eval_temperature=args.eval_temperature,
                goal_info={
                    "goal": successful_goal
                    if successful_goal is not None
                    else badge_goal(observation)
                },
                config={
                    "use_rep": bool(agent.config["use_rep"]),
                    "reset_env": False,
                    "initial_observation": observation,
                    "observation_transform": flatten_observation,
                    "max_episode_steps": chunk_steps,
                    # 평가 구간 도중 파일이 no로 바뀌면 다음 action 전에 중단한다.
                    "continue_condition": lambda: is_agent_enabled(enabled_path),
                    # 환경의 truncated 외에도 badge goal 달성을 즉시 종료 조건으로 쓴다.
                    "stop_condition": final_badge_reached,
                },
            )
            evaluated_steps = env.step_count - start_step
            episode_return = stats.get("final.return", 0.0)
            print(
                f"evaluation step={env.step_count} chunk_steps={evaluated_steps} "
                f"return={episode_return:.3f} badges={tracking_env.last_observation['badges']}"
            )

            # Avoid a busy loop when the toggle changes before a chunk starts.
            if evaluated_steps == 0 and not is_agent_enabled(enabled_path):
                continue

        if env.step_count >= MAX_STEPS:
            print(f"Maximum agent steps reached: {MAX_STEPS}")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        try:
            env.close()
        finally:
            env.pyboy.stop()
        print("PyBoy stopped.")


if __name__ == "__main__":
    main()
