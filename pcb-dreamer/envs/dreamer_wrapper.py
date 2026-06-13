"""Wrapper for DreamerV3 compatibility (old-style 4-return step + dict obs).

Also emits `log_*` keys on each transition. tools.simulate sums these over
an episode and logs the total as a scalar (e.g. `eval_log_failures`), which
is how per-episode PCB diagnostics (routability, length spread, spacing,
invalid actions, reward breakdown) end up in TensorBoard / metrics.jsonl
without any changes to the training loop itself.
"""

import gymnasium.spaces as spaces
import numpy as np
from envs.pcb_env import TPPlacementEnv

# Keys emitted every step. Per-episode metrics (failures, lengths, reward
# breakdown, etc.) are 0 on non-terminal steps and set to their final value
# only on the terminal step, so summing over the episode yields that value.
# `invalid_actions` is emitted as a per-step 0/1 indicator so the episode
# sum yields the total count of invalid placements.
_LOG_KEYS = [
    "log_failures",
    "log_routable",
    "log_total_length",
    "log_length_spread",
    "log_min_tp_spacing",
    "log_invalid_actions",
    "log_reward_routability",
    "log_reward_length",
    "log_reward_spread",
    "log_reward_spacing",
]

_TERMINAL_INFO_KEYS = {
    "log_failures": "failures",
    "log_routable": "routable",
    "log_total_length": "total_length",
    "log_length_spread": "length_spread",
    "log_min_tp_spacing": "min_tp_spacing",
    "log_reward_routability": "reward_routability",
    "log_reward_length": "reward_length",
    "log_reward_spread": "reward_spread",
    "log_reward_spacing": "reward_spacing",
}


class PCBDreamerEnv:
    metadata = {}

    def __init__(self, num_traces=8, seed=0, reward_version="v1"):
        self._inner = TPPlacementEnv(num_traces=num_traces, seed=seed,
                                     reward_version=reward_version)
        self._seed = seed
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return spaces.Dict({
            "image": spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            "is_first": spaces.Box(0, 1, (), dtype=np.uint8),
            "is_last": spaces.Box(0, 1, (), dtype=np.uint8),
            "is_terminal": spaces.Box(0, 1, (), dtype=np.uint8),
            "action_mask": spaces.Box(0, 1, (self._inner.num_candidates,), dtype=np.float32),
        })

    @property
    def action_space(self):
        space = spaces.Box(low=0, high=1,
                           shape=(self._inner.num_candidates,),
                           dtype=np.float32)
        space.discrete = True
        space.n = self._inner.num_candidates
        return space

    def reset(self):
        obs, _ = self._inner.reset(seed=self._seed)
        self._seed += 1
        out = {
            "image": obs, "is_first": True, "is_last": False, "is_terminal": False,
            "action_mask": self._inner.candidate_mask.astype(np.float32),
        }
        out.update({k: 0.0 for k in _LOG_KEYS})
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = self._inner.step(int(action))
        done = terminated or truncated
        out = {
            "image": obs, "is_first": False, "is_last": done, "is_terminal": terminated,
            "action_mask": self._inner.candidate_mask.astype(np.float32),
        }
        out["log_invalid_actions"] = float(info.get("invalid_this_step", False))
        for log_key, info_key in _TERMINAL_INFO_KEYS.items():
            out[log_key] = float(info.get(info_key, 0.0))
        return out, np.float32(reward), done, info

    def render(self):
        return self._inner.render()

    def close(self):
        pass