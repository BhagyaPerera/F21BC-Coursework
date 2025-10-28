from typing import Sequence

"""
Regression loss function: Mean Absolute Error (MAE).
a mathematical measure of how far  predictions are from the actual target values.

This is what the loss function does:
It compares y_predicted (model’s outputs) with y_true (the real values) and gives a single number — the error.
"""

#region Mean absolute error
def mean_absolute_error(y_true: Sequence[float], y_predicted: Sequence[float]) -> float:
    n =len(y_true)   #number of labels
    err=0
    if n > 0:
        for true_val, predicted_val in zip(y_true, y_predicted):
            err+=abs(true_val - predicted_val)                                    #calculate abs error for each value
        return err/n                                                              #return average abs error
    else:
        return 0.0

#endregion

