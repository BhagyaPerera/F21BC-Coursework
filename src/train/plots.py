# src/train/plots.py

from typing import Sequence, List
import matplotlib.pyplot as plt


def plot_convergence(history: Sequence[float]) -> None:
    """
    Plot PSO (or training) best fitness vs iteration.
    history[i] = best fitness after iteration i
    """
    if not history:
        return

    plt.figure()
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness (MAE)")
    plt.title("PSO Convergence")
    plt.grid(True)
    plt.show()


def plot_actual_vs_predicted(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Actual vs Predicted",
) -> None:
    """
    Scatter plot to compare true target values with model predictions.
    Perfect predictions should lie on the y = x line.
    """
    if not y_true or not y_pred:
        return

    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)

    # draw y = x reference line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.grid(True)
    plt.show()


def plot_errors(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Absolute Errors",
) -> None:
    """
    Bar plot of absolute error per sample.
    Useful to see which samples are hard for the model.
    """
    if not y_true or not y_pred:
        return

    errors: List[float] = [abs(t - p) for t, p in zip(y_true, y_pred)]

    plt.figure()
    plt.bar(range(len(errors)), errors)
    plt.xlabel("Sample index")
    plt.ylabel("Absolute error")
    plt.title(title)
    plt.show()


def plot_residuals(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Residuals (actual - predicted)",
) -> None:
    """
    Scatter residual plot:
      x-axis: predicted
      y-axis: residual = actual - predicted
    Good models have residuals scattered around 0.
    """
    if not y_true or not y_pred:
        return

    residuals: List[float] = [t - p for t, p in zip(y_true, y_pred)]

    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_error_boxplot(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_true2: Sequence[float],
    y_pred2: Sequence[float],
) -> None:
    """
    Compare error distributions for two sets, e.g. train vs test.

    y_true/y_pred  -> first box (Train)
    y_true2/y_pred2 -> second box (Test)
    """
    if not y_true or not y_pred or not y_true2 or not y_pred2:
        return

    errors1: List[float] = [abs(t - p) for t, p in zip(y_true, y_pred)]
    errors2: List[float] = [abs(t - p) for t, p in zip(y_true2, y_pred2)]

    plt.figure()
    plt.boxplot([errors1, errors2], labels=["Train", "Test"])
    plt.ylabel("Absolute error")
    plt.title("Error distribution: Train vs Test")
    plt.show()
