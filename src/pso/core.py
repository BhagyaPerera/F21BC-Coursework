# pso/core.py
from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import random
import math

from .config import PSOConfig
from .topology import build_informants_random_k, rebuild_if_needed, build_informants_ring
from .boundary import apply_boundary, clamp_velocity


class PSO:
    """
    Informants PSO following Algorithm 39 (Luke, Essentials of Metaheuristics).
    Comments indicate the corresponding pseudocode line numbers.
    """

    def __init__(self, dimension: int, fitness_fn: Callable[[List[float]], float], config: PSOConfig = PSOConfig()):
        self.dimension = dimension                       #dimensions
        self.fitness_fn = fitness_fn                     #fitness function
        self.config = config                                #PSO_configuration
        self.rng = random.Random()

        # line 7: P ← {}
        self.P: List[List[float]] = []                   #position vector
        self.V: List[List[float]] = []                   #velocity vector

        # personal bests
        self.pbest_pos: List[List[float]] = []           #personal best position
        self.pbest_val: List[float] = []                 #Personal best value

        # line 10: Best* ← □
        self.gbest_pos: Optional[List[float]] = None      #global best position
        self.gbest_val: Optional[float] = None            #global best value


        self.informants: List[List[int]] = []             # informants for each particle

        self.init_swarm()

    #region Initialization
    # ---------- Initialization (lines 8–9) ----------
    def rand_vec(self, low: float, high: float) -> List[float]:
        return [self.rng.uniform(low, high) for _ in range(self.dimension)]

    def init_swarm(self):
        # line 8–9: create random particles with random initial velocities
        if self.config.bounds is None:
            low, high = -0.5, 0.5
        else:
            low, high = self.config.bounds

        #create random position vector
        self.P = [self.rand_vec(low, high) for _ in range(self.config.swarm_size)]

        #create random velocity vector with range of +-10 within boundaries
        span = (high - low) * 0.1
        self.V = [self.rand_vec(-span, span) for _ in range(self.config.swarm_size)]

        # personal bests start at initial positions
        self.pbest_pos = [p[:] for p in self.P]
        self.pbest_val = [math.inf if self.config.minimize else -math.inf
                          for _ in range(self.config.swarm_size)]

        # initial random-k informants (always includes self)
        top = self.config.topology or "random_k"

        # DEFAULT BEHAVIOUR
        if top is None or top == "random_k":
            # Local random-k neighbourhood (default)
            self.informants = build_informants_random_k(
                n=self.config.swarm_size,
                k=self.config.k_informants,
                rng=self.rng
            )

        elif top == "ring":
            # Structured ring neighbourhood
            self.informants = build_informants_ring(
                n=self.config.swarm_size,
                radius=self.config.ring_radius
            )

        elif top == "gbest":
            # Fully connected: every particle informs every other
            self.informants = [
                list(range(self.config.swarm_size))
                for _ in range(self.config.swarm_size)
            ]

        elif top == "fully_random":
            # Fully random neighbourhood every initial build
            self.informants = build_informants_random_k(
                n=self.config.swarm_size,
                k=self.config.swarm_size,  # all particles
                rng=self.rng
            )

        else:
            raise ValueError(f"Unknown topology type: {top}")

    #endregion

    #region utility
    #is_better compare new fitness value with previous fitness value.
    def is_better(self, a: float, b: float) -> bool:
        return (a < b) if self.config.minimize else (a > b)
    #endregion

    #region PSO run
    # ---------- Main loop ----------
    def run(self, verbose: bool = True) -> Tuple[List[float], float, List[float]]:

        best_fitness_history: List[float] = []

        # line 11: repeat
        for iteration in range(self.config.iterations):

            # Optional: dynamic rewire of informants (not in pseudocode; “going further”)
            maybe = rebuild_if_needed(
                iteration, self.config.rewire_every, build_informants_random_k,
                n=self.config.swarm_size, k=self.config.k_informants, rng=self.rng
            )
            if maybe is not None:
                self.informants = maybe

            # lines 12–15: assess fitness, update personal and global bests
            for j in range(self.config.swarm_size):
                x = self.P[j]

                # Invisible boundary: penalize if out of [low,high]
                if self.config.boundary_mode.lower() == "invisible" and self.config.bounds is not None:
                    low, high = self.config.bounds
                    out = any((xi < low or xi > high) for xi in x)
                    f = (math.inf if self.config.minimize else -math.inf) if out else self.fitness_fn(x)
                else:
                    f = self.fitness_fn(x)  # lines 12–13: AssessFitness(x)

                # line 14: update personal best x*
                if self.is_better(f, self.pbest_val[j]):
                    self.pbest_val[j] = f
                    self.pbest_pos[j] = x[:]

                # line 15: update global Best*
                if self.gbest_val is None or self.is_better(f, self.gbest_val):
                    self.gbest_val = f
                    self.gbest_pos = x[:]

            # lines 16–25: determine how to mutate (compute velocity)
            for j in range(self.config.swarm_size):
                x = self.P[j]
                v = self.V[j]

                # line 17: x* <- previous fittest location of x
                x_star = self.pbest_pos[j]

                # line 18: x^+ <- previous fittest location of informants of x (including itself)
                inf_idx = self.informants[j]
                if self.config.minimize:
                    best_inf = min(inf_idx, key=lambda idx: self.pbest_val[idx])
                else:
                    best_inf = max(inf_idx, key=lambda idx: self.pbest_val[idx])
                x_plus = self.pbest_pos[best_inf]

                # line 19: x^! ← previous fittest location of any particle
                x_bang = self.gbest_pos if self.gbest_pos is not None else x_star

                # lines 20–24: update velocity dimension-wise
                for i in range(self.dimension):
                    # line 21: b ∼ U(0, beta)
                    b = self.rng.uniform(0.0, self.config.beta)
                    # line 22: c ∼ U(0, gamma)
                    c = self.rng.uniform(0.0, self.config.gamma)
                    # line 23: d ∼ U(0, delta)
                    d = self.rng.uniform(0.0, self.config.delta)

                    # line 24: v_i ← α v_i + b(x*_i − x_i) + c(x^+_i − x_i) + d(x^!_i − x_i)
                    v[i] = (
                        self.config.alpha * v[i]
                        + b * (x_star[i] - x[i])
                        + c * (x_plus[i] - x[i])
                        + d * (x_bang[i] - x[i])
                    )

                # (practical) velocity clamp
                clamp_velocity(v, self.config.v_clamp)
                self.V[j] = v

            # lines 25–26: move: x ← x + e v  (Mutate)
            for j in range(self.config.swarm_size):
                self.P[j] = [self.P[j][i] + self.config.e * self.V[j][i] for i in range(self.dimension)]
                apply_boundary(self.P[j], self.config.bounds, self.config.boundary_mode)

            # ----- record history here -----
            if self.gbest_val is not None:
                best_fitness_history.append(self.gbest_val)

            # line 27: termination check (time/ideal). Here: iterations.
            if verbose and (iteration % max(1, self.config.iterations // 10) == 0 or iteration == self.config.iterations - 1):
                g = self.gbest_val
                print(f"[PSO] iter {iteration+1}/{self.config.iterations} gbest={g:.6f}" if g is not None else f"[PSO] iter {iteration+1}")

        # line 28: return Best*
        assert self.gbest_pos is not None and self.gbest_val is not None
        return self.gbest_pos, self.gbest_val,best_fitness_history
    #endregion
