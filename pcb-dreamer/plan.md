# Plan: World-Model RL for PCB Test Point Placement

## What This Document Is

A complete implementation plan for training a DreamerV3 agent to automatically place test points on PCB boards for SI (Signal Integrity) test fixtures. This document contains everything needed to build the project from scratch.

---

## 1. The Problem

### Input (from CSV/Excel, varies per board)
- Board dimensions (e.g., 135mm × 90mm)
- Trace starting points from connector escape routing (up to 20 traces)
- Obstacles: non-routing zones (rectangles), UPTHs (circles), tab pads (rectangles)
- Each trace has a known starting point and breakout length

### What the Agent Decides
Where to place each trace's **test point** on the board. Test points are endpoints where signal integrity measurements are taken. Their positions are NOT given — the agent must choose them.

### Constraints (fixed across all boards)
- Trace width: 0.2286mm
- Trace to board edge: ≥0.26mm (edge-to-edge)
- Trace to trace: ≥1.1mm (edge-to-edge)
- Trace to UPTH: ≥0.7mm
- Trace to tab pad: ≥0.7mm
- Test point to test point: ≥13mm (center-to-center)
- Test point to board edge: ≥14mm
- Test point to connector: ≥3mm

### Objectives
- Minimize total trace length (weight 0.5)
- Maximize trace-to-trace spacing (weight 0.5)
- Length matching: soft constraint (keep lengths similar for post-hoc equalization)

### Example Board (from TE Connectivity data)
```
Board: 135×90mm, origin (0, 98.2)
Non-routing zone: bottom-left (58.294, 108.044), 17.8×6.64mm
UPTH 1: center (58.194, 105.894), ø1.9mm
UPTH 2: center (76.194, 105.894), ø1.9mm
Tab Pad 1: (56.151, 113.346), 1.526×1.216mm
Tab Pad 2: (76.711, 113.346), 1.526×1.216mm
20 traces: starting points at y=107.94 and y=114.74, x from 58.94 to 70.04
Breakout length: 0.8626mm per trace
```

---

## 2. Three-Stage Pipeline

**Stage 1 — Test Point Placement (DreamerV3, learned):**
Agent places test points one at a time by selecting from a pre-computed candidate grid. This is the hard combinatorial problem.

**Stage 2 — Trace Routing (A*, deterministic):**
Given starting points and test point locations, A* finds shortest feasible paths with obstacle avoidance. Called during environment evaluation.

**Stage 3 — Length Equalization (deferred/future work):**
Meander shorter traces to match the longest. Soft constraint during placement keeps lengths similar enough for post-hoc equalization.

---

## 3. Action Space

The 13mm TP-to-TP and 14mm TP-to-edge constraints severely restrict valid positions. On a 135×90mm board, a 6.5mm grid produces ~160 candidate positions. The agent picks one per step.

**Action: Discrete(~160)** — index into pre-computed candidate grid.
This matches the DreamerV3+FR paper which uses Discrete(100).

---

## 4. Observation: 64×64 RGB Image

**We render the board as a 64×64 RGB image**, matching the DreamerV3+FR paper's approach. This is critical because DreamerV3's CNN encoder + LayerNorm + symlog are designed and tested for dense image observations. Sparse vector observations cause NaN (LayerNorm divides by near-zero std on mostly-zero inputs).

### Image Channels (rendered into RGB)
- **Red channel**: obstacles (non-routing zones, UPTHs, tab pads) with clearance buffers
- **Green channel**: placed test points + 13mm exclusion zones + routed traces so far
- **Blue channel**: current trace starting point + target heatmap of valid remaining candidates

The board coordinates are normalized to [0, 64] pixels regardless of physical dimensions.

### Why Images, Not Vectors
The DreamerV3+FR paper uses identical dreamerv3-torch source code with NO modifications to handle NaN. Their setup works because image observations are dense (every pixel has a value). We copy their proven approach exactly.

---

## 5. Reward Design

### Per-Step (after each test point placement)
| Condition | Reward |
|-----------|--------|
| Valid placement (all constraints met) | +1.0 |
| Constraint violation (TP spacing, edge) | -2.0 |
| Preserving future options | +0.3 × (valid_remaining / total_candidates) |

### Episode-End (after all TPs placed + A* routing)
| Metric | Reward |
|--------|--------|
| All traces routable | +10.0 |
| Per unroutable trace | -10.0 |
| Total trace length (minimize) | -0.5 × (total / baseline) |
| Min TP spacing (maximize) | +2.0 × (min_spacing / 13mm) |
| Length spread (soft) | -3.0 × (max - min) / mean |

---

## 6. Project Structure

```
pcb-dreamer/
├── envs/
│   ├── board.py              ← board data, candidate grid, constraints
│   ├── routing.py            ← A* pathfinding
│   ├── pcb_env.py            ← Gymnasium env (renders 64×64 image obs)
│   ├── dreamer_wrapper.py    ← old-gym dict wrapper for DreamerV3
│   ├── visualize.py          ← 2D matplotlib plotting
│   └── wrappers.py           ← FROM dreamerv3-torch
├── dreamer.py                ← FROM dreamerv3-torch
├── models.py                 ← FROM dreamerv3-torch (1 small edit)
├── networks.py               ← FROM dreamerv3-torch (1 small edit)
├── tools.py                  ← FROM dreamerv3-torch (4 small edits)
├── exploration.py            ← FROM dreamerv3-torch (as-is)
├── parallel.py               ← FROM dreamerv3-torch (as-is)
├── configs.yaml              ← our config
├── train.py                  ← entry point
├── test_env.py               ← tests
└── requirements.txt
```

### Approach: Fork dreamerv3-torch Source
Same approach as the DreamerV3+FR paper's repo (github.com/yinqimakeitfun/dreamer-Autorouting). Copy dreamerv3-torch source files (from github.com/NM512/dreamerv3-torch) directly into the project and make modifications in-place. No cloning at runtime, no patches.

---

## 7. Files From dreamerv3-torch

Copy these 7 files from https://github.com/NM512/dreamerv3-torch:

| File | Copy to |
|------|---------|
| dreamer.py | dreamer.py |
| models.py | models.py |
| networks.py | networks.py |
| tools.py | tools.py |
| exploration.py | exploration.py |
| parallel.py | parallel.py |
| envs/wrappers.py | envs/wrappers.py |

---

## 8. Required Modifications to dreamerv3-torch Files

### tools.py — 4 changes needed

**8a. OneHotDist numerical stability (~line 430)**
Problem: softmax precision loss + log(0) on large discrete action spaces.
```python
# REPLACE the __init__ method of class OneHotDist:
def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
    if logits is not None:
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
    if logits is not None and unimix_ratio > 0.0:
        probs = F.softmax(logits, dim=-1)
        probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
        probs = torch.clamp(probs, min=1e-8)
        logits = torch.log(probs)
        super().__init__(logits=logits, probs=None, validate_args=False)
    else:
        super().__init__(logits=logits, probs=probs, validate_args=False)
```

**8b. DiscDist device (~line 460)**
Problem: hardcoded device param mismatches when model moves between CPU/GPU.
```python
# CHANGE:
self.buckets = torch.linspace(low, high, steps=255, device=device)
# TO:
self.buckets = torch.linspace(low, high, steps=255, device=logits.device)
```

**8c. sample_episodes numpy 2.x (~line 350)**
Problem: np.append crashes on numpy 2.x with inhomogeneous arrays.
```python
# REPLACE the else block in sample_episodes:
new_ret = {}
for k, v in episode.items():
    if "log_" not in k:
        chunk = v[index : min(index + possible, total)].copy()
        new_ret[k] = np.concatenate([ret[k], chunk], axis=0)
ret = new_ret
```

**8d. Cache list-to-array conversion (~line 335)**
Problem: in-memory episodes are lists, loaded episodes are arrays.
```python
# ADD after episode selection in sample_episodes:
episode = {k: np.stack(v) if isinstance(v, list) else v for k, v in episode.items()}
```

### networks.py — 1 change

**8e. MLP default device (~line 595)**
```python
# CHANGE: device="cuda"
# TO: device="cpu"
```

### models.py — 1 change

**8f. Optional image preprocessing (~line 182)**
```python
# CHANGE:
obs["image"] = obs["image"] / 255.0
# TO:
if "image" in obs:
    obs["image"] = obs["image"] / 255.0
```

### No changes needed: dreamer.py, exploration.py, parallel.py, envs/wrappers.py

---

## 9. Config (configs.yaml)

Use the **same config as the DreamerV3+FR paper** for the image-based setup. Key settings:

```yaml
defaults:
  steps: 500000
  envs: 1
  action_repeat: 1
  time_limit: 20
  prefill: 2500
  train_ratio: 512
  batch_size: 16
  batch_length: 32
  eval_every: 10000
  eval_episode_num: 10
  video_pred_log: true
  device: 'cuda:0'
  precision: 32
  pretrain: 100

  # Model (matches DreamerV3+FR paper)
  dyn_hidden: 512
  dyn_deter: 512
  dyn_stoch: 32
  dyn_discrete: 32
  units: 512
  act: 'SiLU'
  norm: True                  # SAFE with image obs (dense data)

  # CNN encoder/decoder for 64x64 image
  encoder:
    mlp_keys: '$^'
    cnn_keys: 'image'
    act: 'SiLU'
    norm: True
    cnn_depth: 32
    kernel_size: 4
    minres: 4
    mlp_layers: 5
    mlp_units: 1024
    symlog_inputs: True       # SAFE with image obs

  decoder:
    mlp_keys: '$^'
    cnn_keys: 'image'
    act: 'SiLU'
    norm: True
    cnn_depth: 32
    kernel_size: 4
    minres: 4
    mlp_layers: 5
    mlp_units: 1024
    cnn_sigmoid: False
    image_dist: mse
    vector_dist: symlog_mse
    outscale: 1.0

  actor:
    layers: 2
    dist: 'onehot'            # discrete actions
    std: 'none'
    entropy: 3e-3
    unimix_ratio: 0.01
    lr: 3e-5
    grad_clip: 100.0

  critic:
    layers: 2
    dist: 'symlog_disc'
    slow_target: True
    slow_target_fraction: 0.02
    lr: 3e-5
    grad_clip: 100.0

  reward_head: {layers: 2, dist: 'symlog_disc', loss_scale: 1.0}
  cont_head: {layers: 2, loss_scale: 1.0}

  model_lr: 1e-4
  grad_clip: 1000
  discount: 0.997
  imag_horizon: 15
  imag_gradient: 'reinforce'  # for discrete actions

pcb:
  task: 'pcb_tp_placement'
  time_limit: 20
  action_repeat: 1
  steps: 500000
  size: [64, 64]

debug:
  pretrain: 10
  prefill: 200
  batch_size: 10
  batch_length: 20
  steps: 1000
  eval_every: 500
```

This is essentially the same config as the DreamerV3+FR paper (norm=True, symlog=True, full-size model, CNN encoder). The only differences are: `dist: 'onehot'` + `imag_gradient: 'reinforce'` (discrete actions instead of continuous), and our custom environment.

---

## 10. Environment Implementation Details

### board.py
- `BoardSpec` dataclass: origin, width, height, obstacles, traces
- `load_te_example()`: hardcoded TE example data
- `generate_candidate_grid(board, resolution)`: returns valid TP positions
- `check_tp_spacing(placed_tps, x, y)`: validates 13mm constraint
- Constants: all constraint values

### routing.py
- `RoutingGrid`: discretizes board at 1.0mm resolution, rasterizes obstacles
- `find_path(start, end)`: A* with 8-directional movement
- `rasterize_trace_path()`: blocks routed trace + clearance for subsequent routing
- `route_all_traces(board, test_points)`: routes all traces sequentially

### pcb_env.py
- `TPPlacementEnv(gymnasium.Env)`:
  - Action: `Discrete(num_candidates)` (~162 for TE example at 6.5mm grid)
  - Observation: `Box(0, 255, (64, 64, 3), uint8)` — rendered board image
  - Episode: 20 steps (one per trace), then A* routing + reward
  - `render_obs()`: draws obstacles (red), placed TPs + traces (green), current trace + candidates (blue) onto 64×64 image

### dreamer_wrapper.py
- Wraps TPPlacementEnv for dreamerv3-torch's old gym API
- Returns dict obs: `{"image": uint8_array, "is_first": bool, "is_last": bool, "is_terminal": bool}`

### visualize.py
- `plot_board()`: matplotlib 2D plot of board with obstacles, TPs, traces, candidates

---

## 11. train.py Entry Point

```python
torch.distributions.Distribution.set_default_validate_args(False)
```
1. Parse args (logdir, steps, device, seed)
2. Load config from configs.yaml
3. Create PCB env with DreamerV3 wrappers (OneHotAction, TimeLimit, SelectAction, UUID)
4. Prefill replay buffer with random actions
5. Create Dreamer agent
6. Training loop: eval → train block → checkpoint → repeat
7. Save checkpoints to logdir (Google Drive for Colab persistence)

---

## 12. Running on Colab

1. Upload `pcb-dreamer/` folder to Google Drive
2. Open Colab, select T4 GPU runtime
3. Run:
```python
from google.colab import drive
drive.mount('/content/drive')
!pip install torch ruamel.yaml einops gym numpy matplotlib tensorboard
!cp -r "/content/drive/MyDrive/pcb-dreamer" /content/pcb-dreamer
%cd /content/pcb-dreamer
!python train.py --logdir "/content/drive/MyDrive/pcb_results/run1" --device cuda:0 --steps 500000
```

Estimated training time: 4-8 hours on T4 for 500K steps.
Checkpoints save to Drive automatically; rerun to resume if disconnected.

---

## 13. What Success Looks Like

- `model_loss` is finite (not NaN) from the first log ← this is the first gate
- `image_loss` decreases over training (world model learns board dynamics)
- `train_return` increases from ~-140 baseline (agent learns better placements)
- `eval_return` increases (policy generalizes)
- Visualization: agent's placements have fewer routing failures, shorter traces, better spacing than random

---

## 14. Fallback

If DreamerV3 doesn't converge within 500K steps:
```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```
PPO is guaranteed to work. The poster presents PPO results with DreamerV3 as future work.

---

## 15. Related Work

1. **DreamerV3+FR (Expert Systems with Applications, 2026)** — direct predecessor. World-model RL for PCB routing via FreeRouting. 100 discrete actions, 64×64 image obs, CNN encoder. Our setup mirrors theirs.
2. **Fang et al. (DAC '24)** — obstacle-aware length-matching via DP. Our Stage 3.
3. **Vassallo (DATE '24)** — RL for PCB placement with adaptive rewards.
4. **DreamerV3 (Hafner et al. 2023)** — base architecture.

### The Gap
All prior work assumes fixed endpoints. We address automated test point placement where endpoints are design decisions under coupled constraints.
