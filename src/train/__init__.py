"""
Training package for Task 4:
- loads concrete dataset (8 features + 1 target)
- scales using train stats
- trains ANN via PSO (from src.ann_pso.ann_pso)
- evaluates on train + test
"""
from .pipeline import run_pipeline


__all__ = [
    "run_pipeline",
]
