# pso/topology.py
from __future__ import annotations
from typing import List
import random


"""
 build_informants_random_k(n, k, rng) returns, for every particle in a swarm of size n, a list of k randomly chosen informants (plus itself).
 
For each particle j (0…n−1):
    1.Randomly shuffle all particle indices using rng.
    2.Pick the first k indices as j’s informants.
    3.Ensure j itself is included (so it always considers its own best).
    4.Store that sorted list.

Inputs
       n: number of particles in the swarm.
       k: desired number of informants per particle ([1, n]).
       rng: a random.Random instance (for reproducible randomness).

output 
      returns: list of informants for each particle in swarm
"""

#region random k topology
def build_informants_random_k(n: int, k: int, rng: random.Random) -> List[List[int]]:
    k = max(1, min(k, n))               #ensure informants size is within (1,n)
    groups: List[List[int]] = []
    for j in range(n):
        pool = list(range(n))           #add all the particles to pool
        rng.shuffle(pool)               #randomly shuffle all the particles in swarm
        group = set(pool[:k])           #create informants using first k particles
        group.add(j)                    #include particle itself
        groups.append(sorted(group))    #sort group and add to groupList
    return groups                       #return groups

#endregion



"""
Ring topology:
      Particles are arranged on a circle (indices 0..n-1 wrap around).
      Each particle j listens to neighbors within `radius` steps on both sides,
      plus itself (offset 0). Indices are computed with modulo n.

    For each particle j:
      1) Collect neighbor indices.
      2) Convert to a sorted list for readability and determinism.
      3) Store as j’s informant list.

    Inputs
      n       : Total number of particles in the swarm.
      radius  : Neighborhood radius on each side (≥ 0). 
                - radius = 0 -> informants = [j] (self only)
                - radius = 1 -> informants = [j-1, j, j+1] (mod n)
                - radius = 2 -> informants = [j-2,j-1, j, j+1,j+2] (mod n)
                - larger radius adds farther neighbors; if radius ≥ n, everyone informs everyone.

    Output
      returns : informant groups.
    """

#region ring topology
def build_informants_ring(n: int, radius: int = 1) -> List[List[int]]:
    """Ring topology: each particle informed by neighbors within 'radius' plus self."""
    groups: List[List[int]] = []                         #create an empty array groups
    indices = list(range(n))                             #create an empty list indices
    for j in range(n):
        for distance in range(-radius,radius+1):         #for each particle create informants based on radius and modulo
            indices.append((j+ distance) % n)
        groups.append(sorted(indices))
    return groups

#endregion



""" rebuild is a small scheduler that decides whether to rebuild (re-wire) 
the informant network in PSO on a given iteration. If it’s time, it calls a builder
using either
  build_informants_random_k
  build_informants_ring
and returns the new informants matrix; otherwise it returns None and keep existing one.

Inputs
    1. iteration -current iteration
    2. rewrite_every  -how often to rebuild (iteration schedular frequency)
    3.  builder - a function that constructs an informants topology
    4. *args ,*keyword- arguments aand keywords pass through the builder

"""
#region rebuild informants topology
def rebuild_if_needed(iteration: int, rewire_every: int | None,
                      builder, *args, **kwargs) -> List[List[int]] | None:
    if rewire_every is None:
        return None
    if (iteration + 1) % rewire_every == 0:
        return builder(*args, **kwargs)
    return None

#endregion
