"""Round-robin trace-growth PCB router (MaskablePPO edition).

config     : single dataclass holding every tunable parameter
geometry   : clearance-inflated collision primitives + spatial hash
board      : board / connector / pin / obstacle specification
breakout   : optional deterministic breakout (default: disabled)
env        : gymnasium round-robin growth environment
rendering  : matplotlib board rendering (preview + episode renders)
portfolio  : budget-stratified solution portfolio
explorer   : ForcedExplorer (profile-rotating masked-random episodes)
callbacks  : W&B logging, portfolio harvesting, eval, explorer scheduling
validate   : acceptance tests (zero-violation, self-crossing, reward scale)
train      : training entry point
"""

__version__ = "3.0.0"
