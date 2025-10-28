# pso/__init__.py

"""
Particle Swarm Optimization (PSO) Package
-----------------------------------------
This package implements a modular and extensible Particle Swarm optimization based on Algorithm 39
("Informants PSO") from sean Luke's  : Essentials of metaheuristics*

It provides all the necessary building blocks for applying PSO to real-world
optimization tasks — including configurable parameters, swarm initialization,
velocity and boundary management, and flexible neighborhood (informant) topologies.

Modules
--------
- config.py
    Defines the PSOConfig dataclass that holds PSO hyperparameters such as
    swarm size, coefficients (alpha,beta,gamma,delta,eeta), iteration count, bounds, velocity
    clamping, boundary handling, and whether the goal is minimization or maximization.

- core.py
    Contains the main `PSO` class which implements the core PSO algorithm:
    initialization, fitness evaluation, velocity and position updates, and
    global/personal best tracking.

- topology.py
    Provides functions to build informant topologies (social networks among particles):
      * `build_informants_random_k` — randomly assigns k informants per particle.
      * `build_informants_ring` — arranges particles in a ring where each one
        communicates only with neighbors within a given radius.
    Also includes `rebuild_if_needed` for dynamic re-wiring of informants.

- boundary.py
    Implements practical boundary-handling strategies:
      * `clip` — snap particles to the boundary limits.
      * `reflect` — mirror positions back into range.
      * `invisible` — allow out-of-bounds positions (penalized via fitness).

Usage
-----
Typical workflow:
1. Define a fitness function: `def fitness(x): ...`
2. Configure PSO parameters via `PSOConfig(...)`.
3. Create and run the optimizer:

    from pso import PSO, PSOConfig
    cfg = PSOConfig(iters=100, swarmsize=30, bounds=(-5, 5))
    optimizer = PSO(dim=2, fitness_fn=fitness, cfg=cfg)
    best_pos, best_val = optimizer.run()

4. Retrieve the best solution `(best_pos, best_val)`.

Exports
--------
- `PSO`
- `PSOConfig`
- `build_informants_random_k`
- `build_informants_ring`

Version: 1.0
"""

from .config import PSOConfig
from .core import PSO
from .topology import build_informants_random_k, build_informants_ring

__all__ = [
    "PSO",
    "PSOConfig",
    "build_informants_random_k",
    "build_informants_ring",
]
