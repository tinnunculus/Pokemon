"""Evaluate a pretrained HIQL checkpoint on Pokemon Red.

By default, this script runs 300 independent episodes and regards an episode as
successful when the last badge bit (``badges[-1]``) becomes one.  Each episode
starts from ``init.state`` and is capped at 50,000 agent steps.
"""

import argparse
import csv
import json
import time
import uuid
from functools import partial
from pathlib import Path

import jax
import numpy as np

from env.red_gym_env_v2 import RedGymEnv
from hiql_run_pretrained_interactive import (
    SCRIPT_DIR,
    badge_goal,
    build_agent,
    final_badge_reached,
    load_successful_goal,
    resolve_checkpoint,
    resolve_from_script_dir,
)
from jaxrl_m.evaluation import supply_rng
from main import load_simple_yaml
from models.hiql import flatten_observation


DEFAULT_EPISODES = 300
DEFAULT_MAX_EPISODE_STEPS = 50_000


def _get(config, key, default=None):
    current = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate goal achievement of a pretrained HIQL agent."
    )
    parser.add_argument("--checkpoint", default=None, help="HIQL .msgpack checkpoint.")
    parser.add_argument("--config", default="configs/hiql.yaml")
    parser.add_argument("--goal-trajectory", default=None)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS
    )
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument(
        "--output",
        default="hiql_evaluation_results.csv",
        help="Episode-level CSV output path.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1_000,
        help="Print in-episode progress every N agent steps.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than 0")
    if args.max_episode_steps <= 0:
        raise ValueError("--max-episode-steps must be greater than 0")
    if args.eval_temperature < 0:
        raise ValueError("--eval-temperature must be greater than or equal to 0")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be greater than 0")


def select_action(
    agent,
    policy_fn,
    high_policy_fn,
    observation,
    goal,
    eval_temperature,
    discrete,
):
    """Select one action using the same HIQL hierarchy as the interactive runner."""
    flat_observation = flatten_observation(observation)

    if agent.config["use_waypoints"]:
        current_goal = high_policy_fn(
            observations=flat_observation,
            goals=goal,
            temperature=eval_temperature,
        )
        if agent.config["use_rep"]:
            norm = np.linalg.norm(current_goal, axis=-1, keepdims=True)
            current_goal = current_goal / np.maximum(norm, 1e-8)
            current_goal = current_goal * np.sqrt(current_goal.shape[-1])
        else:
            current_goal = flat_observation + current_goal
        low_dim_goals = True
    else:
        if agent.config["use_rep"]:
            current_goal = agent.get_policy_rep(
                targets=goal, bases=flat_observation
            )
            low_dim_goals = True
        else:
            current_goal = goal
            low_dim_goals = False

    action = policy_fn(
        observations=flat_observation,
        goals=current_goal,
        low_dim_goals=low_dim_goals,
        temperature=eval_temperature,
        discrete=discrete,
    )
    return int(np.asarray(action).item())


def run_episode(
    env,
    agent,
    policy_fn,
    high_policy_fn,
    goal,
    max_episode_steps,
    eval_temperature,
    discrete,
    episode,
    total_episodes,
    log_interval,
):
    print(f"[episode {episode}/{total_episodes}] resetting environment...", flush=True)
    reset_start = time.perf_counter()
    observation, _ = env.reset()
    reset_time = time.perf_counter() - reset_start
    episode_return = 0.0

    print(
        f"[episode {episode}/{total_episodes}] started "
        f"(reset={reset_time:.1f}s, max_steps={max_episode_steps:,})",
        flush=True,
    )

    if final_badge_reached(observation):
        return {"success": True, "steps": 0, "return": 0.0}

    for step in range(1, max_episode_steps + 1):
        action = select_action(
            agent,
            policy_fn,
            high_policy_fn,
            observation,
            goal,
            eval_temperature,
            discrete,
        )

        observation, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)

        success = final_badge_reached(observation)
        if step % log_interval == 0:
            badges = np.asarray(observation["badges"]).tolist()
            x_pos, y_pos, map_id = env.get_game_coords()
            print(
                f"[episode {episode}/{total_episodes} | "
                f"step {step:,}/{max_episode_steps:,}] "
                f"return={episode_return:.3f} badges={badges} "
                f"position=(x={x_pos}, y={y_pos}, map={map_id})",
                flush=True,
            )

        if success or terminated or truncated:
            return {
                "success": success,
                "steps": step,
                "return": episode_return,
            }

    return {
        "success": final_badge_reached(observation),
        "steps": max_episode_steps,
        "return": episode_return,
    }


def write_results(output_path, rows, summary):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=("episode", "success", "steps", "return")
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return summary_path


def main():
    startup_start = time.perf_counter()
    args = parse_args()
    validate_args(args)

    config_path = resolve_from_script_dir(args.config)
    config = load_simple_yaml(config_path)
    checkpoint_path = resolve_checkpoint(args.checkpoint, config)
    output_path = resolve_from_script_dir(args.output)
    seed = int(_get(config, "training.seed", 0))
    discrete = int(_get(config, "model.discrete", 1))

    session_path = SCRIPT_DIR / f"evaluation_{str(uuid.uuid4())[:8]}"
    env_config = {
        "headless": True,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str(SCRIPT_DIR / "init.state"),
        "max_steps": args.max_episode_steps,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "session_path": session_path,
        "gb_path": str(SCRIPT_DIR / "red_rom" / "PokemonRed.gb"),
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "extra_buttons": False,
    }

    print("[startup] Initializing environment...", flush=True)
    stage_start = time.perf_counter()
    env = RedGymEnv(env_config)
    print(
        f"[startup] Environment initialized in "
        f"{time.perf_counter() - stage_start:.1f}s",
        flush=True,
    )
    try:
        stage_start = time.perf_counter()
        print("[startup] Resetting environment and loading checkpoint...", flush=True)
        initial_observation, _ = env.reset()
        agent = build_agent(config, initial_observation, checkpoint_path)
        print(
            f"[startup] Checkpoint loaded in "
            f"{time.perf_counter() - stage_start:.1f}s",
            flush=True,
        )

        stage_start = time.perf_counter()
        print("[startup] Loading evaluation goal...", flush=True)
        dataset_dir = resolve_from_script_dir(
            _get(config, "dataset.path", "offline_trajectories")
        )
        goal = load_successful_goal(dataset_dir, args.goal_trajectory)
        if goal is None:
            print(
                "No successful trajectory found; using the initial observation "
                "with badges[-1] set to 1 as the goal."
            )
            goal = badge_goal(initial_observation)
        print(
            f"[startup] Evaluation goal loaded in "
            f"{time.perf_counter() - stage_start:.1f}s",
            flush=True,
        )

        policy_fn = supply_rng(
            partial(agent.sample_actions), rng=jax.random.PRNGKey(seed)
        )
        high_policy_fn = supply_rng(
            agent.sample_high_actions, rng=jax.random.PRNGKey(seed + 1)
        )

        print(f"Checkpoint: {checkpoint_path}")
        print(
            f"Evaluating {args.episodes} episodes "
            f"(max {args.max_episode_steps} steps each, "
            f"logging every {args.log_interval} steps)"
        )
        print(
            f"[startup] Ready in {time.perf_counter() - startup_start:.1f}s. "
            "The first policy call may include JAX compilation time.",
            flush=True,
        )

        rows = []
        evaluation_start = time.perf_counter()
        for episode in range(1, args.episodes + 1):
            result = run_episode(
                env=env,
                agent=agent,
                policy_fn=policy_fn,
                high_policy_fn=high_policy_fn,
                goal=goal,
                max_episode_steps=args.max_episode_steps,
                eval_temperature=args.eval_temperature,
                discrete=discrete,
                episode=episode,
                total_episodes=args.episodes,
                log_interval=args.log_interval,
            )
            row = {"episode": episode, **result}
            rows.append(row)

            successes = sum(int(item["success"]) for item in rows)
            print(
                f"[episode {episode}/{args.episodes}] complete "
                f"steps={result['steps']:,} return={result['return']:.3f} "
                f"goal_reached={result['success']} total_success={successes} "
                f"rate={successes / episode:.2%}",
                flush=True,
            )

        success_count = sum(int(row["success"]) for row in rows)
        summary = {
            "checkpoint": str(checkpoint_path),
            "episodes": args.episodes,
            "success_count": success_count,
            "failure_count": args.episodes - success_count,
            "success_rate": success_count / args.episodes,
            "average_return": float(np.mean([row["return"] for row in rows])),
            "average_steps": float(np.mean([row["steps"] for row in rows])),
            "max_episode_steps": args.max_episode_steps,
            "eval_temperature": args.eval_temperature,
            "success_condition": "badges[-1] == 1",
        }
        summary_path = write_results(output_path, rows, summary)

        print("\nEvaluation complete")
        print(f"Evaluation time: {time.perf_counter() - evaluation_start:.1f}s")
        print(
            f"Episodes that reached the goal: "
            f"{success_count}/{args.episodes}"
        )
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Average return: {summary['average_return']:.3f}")
        print(f"Average steps: {summary['average_steps']:.1f}")
        print(f"Episode results: {output_path}")
        print(f"Summary: {summary_path}")
    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")
    finally:
        try:
            env.close()
        finally:
            env.pyboy.stop()


if __name__ == "__main__":
    main()
