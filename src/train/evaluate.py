from typing import Sequence, List

from src.ann.network import Network
from src.ann.loss import mean_absolute_error


#region evaluate
def evaluate_mae(
    net: Network,
    x: Sequence[Sequence[float]],
    y_true: Sequence[float],
) -> float:
    predictions: List[float] = []
    for row in x:
        out = net.forward_algorithm(row)  # call to forward algorithm in ann
        predictions.append(out[0])
    return mean_absolute_error(list(y_true), predictions)   #call to mean_absolute_error function in ann.loss
#endregion
