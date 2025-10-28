# pso/config.py
"""Define all configurable parameters used by PSO"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class PSOConfig:
    # Algorithm 39 coefficients (pseudocode lines 1–6)
    swarm_size: int = 40         #number of particles
    alpha: float = 0.72          #proportion of velocity to be retained
    beta: float = 1.49           #proportion of personal best to be retained
    gamma: float = 1.49          #proportion of the informants’ best to be retained
    delta: float = 0.0           #proportion of global best to be retained
    e: float = 1.0               #jump_size of a particle

    # Practical controls (run settings) (termination, init, etc.)
    iterations: int = 300
    seed: Optional[int] = 42
    minimize: bool = True

    # Initialization / search space
    bounds: Optional[Tuple[float, float]] = (-2.0, 2.0)
    v_clamp: Optional[Tuple[float, float]] = None

    # Topology
    k_informants: int = 3        # random-k informants (includes self)
    rewire_every: Optional[int] = None  # e.g. 25 to refresh informants every 25 iters

    # Boundary handling: 'clip', 'reflect', 'invisible'
    boundary_mode: str = "clip"
