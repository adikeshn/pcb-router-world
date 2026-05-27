"""Wrapper for DreamerV3 compatibility (old gym API + dict obs)."""

import gym
import gym.spaces
import numpy as np
from envs.pcb_env import TPPlacementEnv


class PCBDreamerEnv:
    metadata = {}

    def __init__(self, num_traces=8, seed=0):
        self._inner = TPPlacementEnv(num_traces=num_traces)
        self._seed = seed
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.uint8),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.uint8),
        })

    @property
    def action_space(self):
        space = gym.spaces.Discrete(self._inner.num_candidates)
        space.discrete = True
        return space

    def reset(self):
        obs, _ = self._inner.reset(seed=self._seed)
        self._seed += 1
        return {"image": obs, "is_first": True, "is_last": False, "is_terminal": False}

    def step(self, action):
        obs, reward, terminated, truncated, info = self._inner.step(int(action))
        done = terminated or truncated
        return (
            {"image": obs, "is_first": False, "is_last": done, "is_terminal": terminated},
            np.float32(reward), done, info,
        )

    def render(self):
        return self._inner.render()

    def close(self):
        pass