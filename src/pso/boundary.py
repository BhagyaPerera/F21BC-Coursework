# pso/boundary.py
from __future__ import annotations
from typing import List, Optional, Tuple


"""
apply_boundary():
  Keeps every particle inside the allowed search space
  Prevents invalid calculations
  Helps the algorithm remain stable and realistic
Inputs
  1.x: the particle’s position vector.
  2.bounds: (low, high) numeric range applied to every dimension. If None, no bounding.
  3.mode: how to handle coordinates that fall outside [low, high]."""

#region apply boundary
def apply_boundary(x: List[float],
                   bounds: Optional[Tuple[float, float]],
                   mode: str) -> None:
    if bounds is None:            #if there are no boundaries defined return none.
        return
    low, high = bounds            #assigned bounds to low,high
    m = mode.lower()              #convert mode to lower

    #region clip - clamps to edges
    if m == "clip":
        for i in range(len(x)):
            if x[i] < low:
                  x[i] = low        #if current position is lower than lower bound assigned lower bound position
            elif x[i] > high:
                  x[i] = high       #if current position is higher than higher bound assigned higher bound postion
    #endregion

    #region reflects - make mirror position later clamp
    elif m == "reflect":
        for i in range(len(x)):

            if x[i] < low:         #if position is lower than lower bound
                x[i] = low + (low - x[i])
            if x[i] > high:         #if position is higher than higher bound
                x[i] = high - (x[i] - high)

            #final clamp
            if x[i] < low:
                x[i] = low
            if x[i] > high:
                 x[i] = high
    #endregion

    #region invisible -Does nothing. rely on fitness penalty
    elif m == "invisible":
        pass
    #endregion

    #region invalid mode - if invalid mode throw an error
    else:
        raise ValueError("Unknown boundary_mode (use 'clip', 'reflect', 'invisible').")
    #endregion

#endregion




"""
 clamp_velocity: In PSO, velocity is propotional to step. 
 clamp velocity keep those steps in reasonable limits
 
 Inputs
   1.v:particle's position vector.
   2.v_clamp: boundary limits
"""

#region clamp velocity
def clamp_velocity(v: List[float], v_clamp: Optional[Tuple[float, float]]) -> None:
    if v_clamp is None:
        return
    low, high = v_clamp
    for i in range(len(v)):
        if v[i] < low:
            v[i] = low
        elif v[i] > high:
            v[i] = high

#endregion
