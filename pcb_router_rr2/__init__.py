"""Round-robin trace-growth PCB router (MaskablePPO, fixed budget).

config     : every tunable parameter
geometry   : clearance-inflated collision primitives + spatial hash
board      : board / connector / pin / obstacle specification
breakout   : optional deterministic breakout (default: disabled)
env        : gymnasium round-robin growth environment
rendering  : board rendering (preview, episode-stamped renders)
portfolio  : diverse top-K solution portfolio
explorer   : ForcedExplorer (masked-random episodes)
callbacks  : W&B logging, portfolio, eval, checkpointing, collapse guard
validate   : synthetic acceptance tests
train      : training entry point
"""

__version__ = "5.0.0"
