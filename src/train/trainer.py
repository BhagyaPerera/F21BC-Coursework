from typing import Sequence, Tuple, List

from src.ann.network import Network
from src.ann_pso.ann_pso import train_ann_with_pso

"""
Thin wrapper so other modules don't need to import ann_pso directly.
Returns:
  - trained Network (with best params loaded)
  - best fitness (MAE) from PSO
"""
#region trainer
def train_with_pso(
    x_train: Sequence[Sequence[float]],
    y_train: Sequence[float],
) -> Tuple[Network, float, List[float]]:
    net, best_fit, history = train_ann_with_pso(x_train, y_train)
    return net, best_fit, history

#endregion

