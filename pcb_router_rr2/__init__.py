"""Round-robin trace-growth PCB router, v2 (MaskablePPO edition).

Modules
-------
config     : single dataclass holding every tunable parameter
geometry   : clearance-inflated collision primitives + spatial hash
board      : board / connector / pin / obstacle specification
breakout   : deterministic two-phase, length-normalised breakout
env        : gymnasium round-robin growth environment
rendering  : matplotlib board rendering (preview + episode renders)
portfolio  : diverse top-K solution portfolio
explorer   : ForcedExplorer (masked-uniform-random full episodes)
callbacks  : W&B logging, portfolio harvesting, eval, explorer scheduling
validate   : acceptance tests (zero-violation check, reward-scale check)
train      : training entry point
"""

__version__ = "2.0.0"
