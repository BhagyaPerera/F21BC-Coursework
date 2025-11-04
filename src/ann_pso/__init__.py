"""
PSO has many particles.
 each particle = one set of ANN weights/biases.
 For each particle, PSO calls the fitness function.
 The fitness function does:
     put particle’s weights into the ANN,
     run ANN on all samples,
     compare with real outputs,
     return the error (MAE)."""

from .ann_pso import make_ann_objective,train_ann_with_pso


__all__ = [
    "make_ann_objective",
    "train_ann_with_pso",
]
