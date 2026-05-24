"""Quick test: verify environment and image rendering work."""
import numpy as np
from envs.pcb_env import TPPlacementEnv
import matplotlib.pyplot as plt

env = TPPlacementEnv(num_traces=8)
obs, info = env.reset(seed=42)
print(f"Obs: {obs.shape}, dtype={obs.dtype}, range=[{obs.min()}, {obs.max()}]")
print(f"Action space: Discrete({env.num_candidates})")
print(f"Traces: {env.num_traces}")

# Save initial observation
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(obs); axes[0].set_title("Combined")
axes[1].imshow(obs[:,:,0], cmap='Reds'); axes[1].set_title("R: Obstacles")
axes[2].imshow(obs[:,:,1], cmap='Greens'); axes[2].set_title("G: Placed TPs")
axes[3].imshow(obs[:,:,2], cmap='Blues'); axes[3].set_title("B: Current+Candidates")
for ax in axes: ax.axis('off')
plt.savefig("obs_initial.png", dpi=100, bbox_inches='tight')
print("Saved obs_initial.png")

# Run random episode
total_reward = 0
for step in range(8):
    valid = np.where(env.candidate_mask)[0]
    action = np.random.choice(valid) if len(valid) > 0 else env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    if done: break

print(f"Episode: {step+1} steps, reward={total_reward:.1f}")
print(f"Info: {info}")

# Save final observation
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(obs); axes[0].set_title("Combined")
axes[1].imshow(obs[:,:,0], cmap='Reds'); axes[1].set_title("R: Obstacles")
axes[2].imshow(obs[:,:,1], cmap='Greens'); axes[2].set_title("G: Placed TPs")
axes[3].imshow(obs[:,:,2], cmap='Blues'); axes[3].set_title("B: Current")
for ax in axes: ax.axis('off')
plt.savefig("obs_final.png", dpi=100, bbox_inches='tight')
print("Saved obs_final.png")

# Gym check
from gymnasium.utils.env_checker import check_env
check_env(TPPlacementEnv(num_traces=8), skip_render_check=True)
print("Gymnasium check: PASSED")
