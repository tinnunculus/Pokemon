from typing import Dict
import jax
import numpy as np
from collections import defaultdict
import time
import gymnasium as gym


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wrapper that supplies a jax random key to a function (using keyword `seed`).
    Useful for stochastic policies that require randomness.

    Similar to functools.partial(f, seed=seed), but makes sure to use a different
    key for each new call (to avoid stale rng keys).

    """

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.8, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img


def evaluate_with_trajectories(
        policy_fn, high_policy_fn, policy_rep_fn, env: gym.Env, env_name, num_episodes: int, base_observation=None, num_video_episodes=0,
        use_waypoints=False, eval_temperature=0, epsilon=0, goal_info=None,
        config=None,
) -> Dict[str, float]:
    """Policy를 환경에서 실행하면서 통계와 transition trajectory를 수집한다.

    이 함수는 학습 weight를 갱신하지 않는다. 현재 policy가 실제 환경에서
    어떤 action을 선택하고 어떤 상태로 이동하는지 rollout하여 다음을 반환한다.

    * stats: info 값의 평균과 마지막 episode return
    * trajectories: observation/action/reward/done/info transition 목록
    * renders: video episode를 요청했을 때 수집한 frame

    Pokemon runner는 reset_env=False와 max_episode_steps=100을 전달하므로,
    동일한 PyBoy game을 유지하면서 100-step 단위로 이 함수를 반복 호출한다.
    """
    config = {} if config is None else config
    trajectories = []
    stats = defaultdict(list)

    renders = []
    for i in range(num_episodes + num_video_episodes):
        trajectory = defaultdict(list)

        if 'procgen' in env_name:
            from src.envs.procgen_env import ProcgenWrappedEnv
            from src.envs.procgen_viz import ProcgenLevel
            eval_level = goal_info['eval_level']
            cur_level = eval_level[np.random.choice(len(eval_level))]

            level_details = ProcgenLevel.create(cur_level)
            border_states = [i for i in range(len(level_details.locs)) if len([1 for j in range(len(level_details.locs)) if abs(level_details.locs[i][0] - level_details.locs[j][0]) + abs(level_details.locs[i][1] - level_details.locs[j][1]) < 7]) <= 2]
            target_state = border_states[np.random.choice(len(border_states))]
            goal_img = level_details.imgs[target_state]
            goal_loc = level_details.locs[target_state]
            env = ProcgenWrappedEnv(1, 'maze', cur_level, 1)

        if config.get('reset_env', True):
            observation = env.reset()
            # Gymnasium reset returns ``(observation, info)``.
            if isinstance(observation, tuple) and len(observation) == 2:
                observation = observation[0]
        else:
            observation = config['initial_observation']
        done = False

        # Set goal
        if 'antmaze' in env_name:
            goal = env.wrapped_env.target_goal
            obs_goal = base_observation.copy()
            obs_goal[:2] = goal
        elif 'kitchen' in env_name:
            observation, obs_goal = observation[:30], observation[30:]
            obs_goal[:9] = base_observation[:9]
        elif 'calvin' in env_name:
            observation = observation['ob']
            goal = np.array([0.25, 0.15, 0, 0.088, 1, 1])
            obs_goal = base_observation.copy()
            obs_goal[15:21] = goal
        elif 'procgen' in env_name:
            from src.envs.procgen_viz import get_xy_single
            observation = observation[0]
            obs_goal = goal_img
        elif 'pokemon' in env_name.lower():
            # Pokemon observation은 여러 배열을 담은 dict이지만 HIQL network는
            # 하나의 vector를 입력받으므로 runner가 제공한 함수로 flatten한다.
            observation_transform = config.get('observation_transform', lambda x: x)
            observation = observation_transform(observation)
            obs_goal = goal_info['goal']
        else:
            raise NotImplementedError

        render = []
        step = 0
        info = {}
        while not done:
            continue_condition = config.get('continue_condition')
            if continue_condition is not None and not continue_condition():
                break
            if not use_waypoints:
                # Hierarchy를 쓰지 않을 때는 최종 goal(또는 그 representation)을
                # low-level policy에 직접 전달한다.
                cur_obs_goal = obs_goal
                if config['use_rep']:
                    cur_obs_goal_rep = policy_rep_fn(targets=cur_obs_goal, bases=observation)
                else:
                    cur_obs_goal_rep = cur_obs_goal
            else:
                # HIQL의 high-level actor가 (현재 상태, 최종 goal)을 보고
                # low-level actor가 따라갈 latent waypoint를 매 step 선택한다.
                cur_obs_goal = high_policy_fn(observations=observation, goals=obs_goal, temperature=eval_temperature)
                if config['use_rep']:
                    cur_obs_goal = cur_obs_goal / np.linalg.norm(cur_obs_goal, axis=-1, keepdims=True) * np.sqrt(cur_obs_goal.shape[-1])
                else:
                    cur_obs_goal = observation + cur_obs_goal
                cur_obs_goal_rep = cur_obs_goal

            # low-level actor는 (현재 상태, waypoint)의 categorical distribution을
            # 만들고 실제 환경에 전달할 discrete action 하나를 sample한다.
            action = policy_fn(observations=observation, goals=cur_obs_goal_rep, low_dim_goals=True, temperature=eval_temperature)
            if 'antmaze' in env_name:
                next_observation, r, done, info = env.step(action)
            elif 'kitchen' in env_name:
                next_observation, r, done, info = env.step(action)
                next_observation = next_observation[:30]
            elif 'calvin' in env_name:
                next_observation, r, done, info = env.step({'ac': np.array(action)})
                next_observation = next_observation['ob']
                del info['robot_info']
                del info['scene_info']
            elif 'procgen' in env_name:
                if np.random.random() < epsilon:
                    action = np.random.choice([2, 3, 5, 6])

                next_observation, r, done, info = env.step(np.array([action]))
                next_observation = next_observation[0]
                r = 0.
                done = done[0]
                info = dict()

                loc = get_xy_single(next_observation)
                if np.linalg.norm(loc - goal_loc) < 4:
                    r = 1.
                    done = True

                cur_render = next_observation
            elif 'pokemon' in env_name.lower():
                # JAX scalar를 Python int로 바꿔 PyBoy environment를 한 step 진행한다.
                step_result = env.step(int(np.asarray(action).item()))
                if len(step_result) == 5:
                    raw_next_observation, r, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                else:  # Compatibility with the legacy four-value Gym API.
                    raw_next_observation, r, done, info = step_result
                next_observation = observation_transform(raw_next_observation)

                stop_condition = config.get('stop_condition')
                if stop_condition is not None:
                    done = done or bool(stop_condition(raw_next_observation))

            step += 1

            max_episode_steps = config.get('max_episode_steps')
            if max_episode_steps is not None and step >= max_episode_steps:
                done = True

            # Render
            if 'procgen' in env_name:
                cur_frame = cur_render.transpose(2, 0, 1).copy()
                cur_frame[2, goal_loc[1]-1:goal_loc[1]+2, goal_loc[0]-1:goal_loc[0]+2] = 255
                cur_frame[:2, goal_loc[1]-1:goal_loc[1]+2, goal_loc[0]-1:goal_loc[0]+2] = 0
                render.append(cur_frame)
            else:
                if i >= num_episodes and step % 3 == 0:
                    if 'antmaze' in env_name:
                        size = 200
                        cur_frame = env.render(mode='rgb_array', width=size, height=size).transpose(2, 0, 1).copy()
                        if use_waypoints and not config['use_rep'] and ('large' in env_name or 'ultra' in env_name):
                            def xy_to_pixxy(x, y):
                                if 'large' in env_name:
                                    pixx = (x / 36) * (0.93 - 0.07) + 0.07
                                    pixy = (y / 24) * (0.21 - 0.79) + 0.79
                                elif 'ultra' in env_name:
                                    pixx = (x / 52) * (0.955 - 0.05) + 0.05
                                    pixy = (y / 36) * (0.19 - 0.81) + 0.81
                                return pixx, pixy
                            x, y = cur_obs_goal_rep[:2]
                            pixx, pixy = xy_to_pixxy(x, y)
                            cur_frame[0, int((pixy - 0.02) * size):int((pixy + 0.02) * size), int((pixx - 0.02) * size):int((pixx + 0.02) * size)] = 255
                            cur_frame[1:3, int((pixy - 0.02) * size):int((pixy + 0.02) * size), int((pixx - 0.02) * size):int((pixx + 0.02) * size)] = 0
                        render.append(cur_frame)
                    elif 'kitchen' in env_name:
                        render.append(kitchen_render(env, wh=200).transpose(2, 0, 1))
                    elif 'calvin' in env_name:
                        cur_frame = env.render(mode='rgb_array').transpose(2, 0, 1)
                        render.append(cur_frame)
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=r,
                done=done,
                info=info,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
        if 'calvin' in env_name:
            info['return'] = sum(trajectory['reward'])
        elif 'procgen' in env_name or 'pokemon' in env_name.lower():
            info['return'] = sum(trajectory['reward'])
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)
        if i >= num_episodes:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories, renders


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()
