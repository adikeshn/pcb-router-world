"""
Forced-exploration mode for diverse portfolio population.

DreamerV3's imagination-based training reinforces whatever routes already
worked, causing the policy to concentrate probability on a narrow spatial
region even when entropy is high. The portfolio ends up with only 1-2 distinct
layouts because the learned policy rarely ventures far from its established
solution cluster.

This module addresses that by periodically running PURELY RANDOM episodes
during training -- no learned policy, just masked uniform sampling across the
full candidate grid -- and feeding every valid result directly to the
portfolio tracker. Key properties:

- Random episodes respect the action_mask so they only pick from valid (real,
  unmasked) candidates -- full spatial coverage of the board, not padding-slot
  waste.
- They do NOT go into the DreamerV3 replay buffer so the world model's
  training distribution isn't polluted with random-walk noise.
- Each forced-explore batch runs N_episodes episodes from a fresh env
  instance, so it's fully independent of the main training state.
- The tracker's diversity filter still applies, so the portfolio only fills
  with genuinely distinct placements.

Usage (in the training loop):
    explorer = ForcedExplorer(logdir, board, args)
    # In the training loop, after each eval/train cycle:
    if explorer.should_run(agent._step):
        n_new = explorer.run(tracker, agent._step)
"""

import numpy as np
from envs.dreamer_wrapper import PCBDreamerEnv


class ForcedExplorer:
    """Runs masked-random episodes and feeds valid results to the portfolio."""

    def __init__(self, logdir, make_env_fn, config):
        """
        Args:
            logdir:       Path to run directory (for logging).
            make_env_fn:  Callable that returns a fresh PCBDreamerEnv.
            config:       Namespace with:
                            force_explore_every   (steps between batches)
                            force_explore_episodes (episodes per batch)
                            force_explore_start   (step to begin, default 0)
        """
        self._make_env = make_env_fn
        self._every = getattr(config, "force_explore_every", 1000)
        self._n_eps = getattr(config, "force_explore_episodes", 50)
        self._start = getattr(config, "force_explore_start", 0)
        self._last_ran = -self._every  # ensures first run at step >= _start

    def should_run(self, step: int) -> bool:
        return (step >= self._start and
                step - self._last_ran >= self._every)

    def run(self, tracker, step: int, rng_seed: int = None) -> int:
        """Run a batch of forced-random episodes, feeding valid results to tracker.

        Returns the number of new/improved portfolio slots created.
        """
        rng = np.random.RandomState(rng_seed if rng_seed is not None
                                    else step % (2**31))
        env = self._make_env()
        n_new = 0
        n_routable = 0

        for _ in range(self._n_eps):
            obs = env.reset()
            done = False
            while not done:
                # Masked uniform random: only pick from valid candidates.
                # action_mask is in the obs dict (set up by dreamer_wrapper).
                mask = obs.get("action_mask", None)
                if mask is not None and mask.sum() > 0:
                    probs = mask / mask.sum()
                    action_idx = rng.choice(len(mask), p=probs)
                else:
                    # Fallback: uniform over the action space size.
                    n_act = getattr(env._inner, "num_candidates", None)
                    if n_act is None:
                        n_act = len(mask) if mask is not None else 8
                    action_idx = rng.randint(n_act)
                obs, _reward, done, _info = env.step(action_idx)

            # Feed to tracker (reuses the same diversity filter)
            status = tracker.update(env._inner, step, source="explore")
            if status:
                n_new += 1
            m = env._inner._terminal_metrics
            # Old env reports "failures" (0 = routable); grow env reports
            # "routable" (1.0 = complete + valid). Accept either.
            if m and (m.get("failures", 1) == 0 or m.get("routable", 0.0) == 1.0):
                n_routable += 1

        env.close()
        return n_new, n_routable
