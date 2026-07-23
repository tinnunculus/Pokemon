from pathlib import Path
import numpy as np


def analyze_trajectories(folder):
    folder = Path(folder)
    files = sorted(folder.glob("trajectory_seed_*.npz"))

    if len(files) == 0:
        print("No trajectory files found.")
        return

    total_rewards = []
    max_rewards = []
    mean_rewards = []
    median_rewards = []
    min_rewards = []
    badge_last_is_one = 0

    trajectory_stats = []

    for file in files:
        data = np.load(file)

        rewards = data["rewards"]
        final_badges = data["final_badges"]

        total_reward = rewards.sum()
        max_reward = rewards.max()
        mean_reward = rewards.mean()
        median_reward = np.median(rewards)
        min_reward = rewards.min()

        badge_flag = int(final_badges[-1] == 1)
        badge_last_is_one += badge_flag

        trajectory_stats.append({
            "file": file.name,
            "steps": len(rewards),
            "total_reward": total_reward,
            "max_reward": max_reward,
            "mean_reward": mean_reward,
            "median_reward": median_reward,
            "min_reward": min_reward,
            "badge_last_is_one": badge_flag,
        })

        total_rewards.append(total_reward)
        max_rewards.append(max_reward)
        mean_rewards.append(mean_reward)
        median_rewards.append(median_reward)
        min_rewards.append(min_reward)

    total_rewards = np.array(total_rewards)
    max_rewards = np.array(max_rewards)
    mean_rewards = np.array(mean_rewards)
    median_rewards = np.array(median_rewards)
    min_rewards = np.array(min_rewards)

    print("=" * 60)
    print(f"Number of trajectories : {len(files)}")
    print(f"final_badges[-1] == 1  : {badge_last_is_one}")
    print("=" * 60)

    print("\n=== Total Reward Statistics ===")
    print(f"Mean   : {total_rewards.mean():.3f}")
    print(f"Median : {np.median(total_rewards):.3f}")
    print(f"Max    : {total_rewards.max():.3f}")
    print(f"Min    : {total_rewards.min():.3f}")

    print("\n=== Max Reward Statistics ===")
    print(f"Mean   : {max_rewards.mean():.3f}")
    print(f"Median : {np.median(max_rewards):.3f}")
    print(f"Max    : {max_rewards.max():.3f}")
    print(f"Min    : {max_rewards.min():.3f}")

    print("\n=== Mean Reward Statistics ===")
    print(f"Mean   : {mean_rewards.mean():.3f}")
    print(f"Median : {np.median(mean_rewards):.3f}")
    print(f"Max    : {mean_rewards.max():.3f}")
    print(f"Min    : {mean_rewards.min():.3f}")

    print("\n=== Per-Trajectory Statistics ===")
    for stat in trajectory_stats:
        print(
            f"{stat['file']:30s} "
            f"steps={stat['steps']:4d} "
            f"total={stat['total_reward']:8.2f} "
            f"max={stat['max_reward']:6.2f} "
            f"mean={stat['mean_reward']:6.2f} "
            f"median={stat['median_reward']:6.2f} "
            f"badge={stat['badge_last_is_one']}"
        )

    return trajectory_stats

stats = analyze_trajectories("/home/yeonlife/projects/Pokemon/offline_trajectories")