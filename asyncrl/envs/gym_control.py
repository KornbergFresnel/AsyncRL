from collections import deque

import gym
import numpy as np

from gym import spaces
from gym.spaces.box import Box
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers.transform_reward import TransformReward


def create_gym_control(
    env_id: str, use_reward_clip: bool = False, max_episode_steps: int = 500
):
    env = gym.make(env_id)
    if use_reward_clip:
        env = TransformReward(env, lambda r: 0.1 * r)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
