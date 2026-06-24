"""DreamerV3 wrapper for the trace-growth env.

Mirrors PCBDreamerEnv (old-style 4-return step + dict obs + per-step log_*
keys) but for TraceGrowEnv. Two differences from the placement wrapper:

  * Observation includes a `trace_id` one-hot vector so the world model knows
    which trace is currently active. The encoder picks this up via mlp_keys.
  * `action_mask` has length NUM_DIRECTIONS (8), not MAX_CANDIDATES.

Per-episode metrics are emitted as log_* keys, 0 on non-terminal steps and set
to their final value only on the terminal step, so tools.simulate's episode sum
yields the right scalar in metrics.jsonl / wandb.
"""

import gymnasium.spaces as spaces
import numpy as np

from envs.pcb_grow_env import TraceGrowEnv, NUM_DIRECTIONS

_LOG_KEYS = [
    "log_routable",
    "log_min_tp_spacing",
    "log_total_length",
    "log_length_spread",
    "log_endpoints_valid",
    "log_spacing_ok",
    "log_invalid_actions",
    "log_reward_spacing",
    "log_reward_gate",
]

_TERMINAL_INFO_KEYS = {
    "log_routable": "routable",
    "log_min_tp_spacing": "min_tp_spacing",
    "log_total_length": "total_length",
    "log_length_spread": "length_spread",
    "log_endpoints_valid": "endpoints_valid",
    "log_spacing_ok": "spacing_ok",
    "log_reward_spacing": "reward_spacing",
    "log_reward_gate": "reward_gate",
}


class PCBGrowDreamerEnv:
    metadata = {}

    def __init__(self, num_traces=8, seed=0, max_length_mm=60.0,
                 img_size=128, board_width=135.0, board_height=90.0,
                 step_mm=2.0, trace_indices=None, dense_reward_weight=0.005):
        self._inner = TraceGrowEnv(
            num_traces=num_traces, seed=seed,
            max_length_mm=max_length_mm, img_size=img_size,
            board_width=board_width, board_height=board_height,
            step_mm=step_mm, trace_indices=trace_indices,
            dense_reward_weight=dense_reward_weight,
        )
        self._seed = seed
        self._img_size = img_size
        self._num_traces = self._inner.num_traces
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return spaces.Dict({
            "image": spaces.Box(0, 255, (self._img_size, self._img_size, 3),
                                dtype=np.uint8),
            "trace_id": spaces.Box(0, 1, (self._num_traces,), dtype=np.float32),
            "is_first": spaces.Box(0, 1, (), dtype=np.uint8),
            "is_last": spaces.Box(0, 1, (), dtype=np.uint8),
            "is_terminal": spaces.Box(0, 1, (), dtype=np.uint8),
            "action_mask": spaces.Box(0, 1, (NUM_DIRECTIONS,), dtype=np.float32),
        })

    @property
    def action_space(self):
        space = spaces.Box(low=0, high=1, shape=(NUM_DIRECTIONS,),
                           dtype=np.float32)
        space.discrete = True
        space.n = NUM_DIRECTIONS
        return space

    def reset(self):
        obs, _ = self._inner.reset(seed=self._seed)
        self._seed += 1
        out = {
            "image": obs,
            "trace_id": self._inner._trace_id_onehot(),
            "is_first": True, "is_last": False, "is_terminal": False,
            "action_mask": self._inner.current_mask.astype(np.float32),
        }
        out.update({k: 0.0 for k in _LOG_KEYS})
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = self._inner.step(int(action))
        done = terminated or truncated
        out = {
            "image": obs,
            "trace_id": self._inner._trace_id_onehot(),
            "is_first": False, "is_last": done, "is_terminal": terminated,
            "action_mask": self._inner.current_mask.astype(np.float32),
        }
        out["log_invalid_actions"] = float(info.get("invalid_this_step", False))
        for log_key, info_key in _TERMINAL_INFO_KEYS.items():
            out[log_key] = float(info.get(info_key, 0.0))
        return out, np.float32(reward), done, info

    def render(self):
        return self._inner.render()

    def close(self):
        pass
