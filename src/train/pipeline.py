from pathlib import Path

from src.train import config
from src.train.data import load_data, train_test_split
from src.train.preprocess import fit_minmax, transform_minmax
from src.train.trainer import train_with_pso
from src.train.evaluate import evaluate_mae


# src/train/pipeline.py
from pathlib import Path
from typing import Sequence, List, Dict, Any

from src.train import config
from src.train.data import load_data, train_test_split
from src.train.preprocess import fit_minmax, transform_minmax
from src.train.trainer import train_with_pso
from src.train.evaluate import evaluate_mae
from src.train import plots  # 👈 make sure you created plots.py
from src.ann.network import Network


def predict_all(net: Network, x: Sequence[Sequence[float]]) -> List[float]:
    predictions: List[float] = []
    for row in x:
        out = net.forward_algorithm(row)
        predictions.append(out[0])
    return predictions


def run_single_training(return_history_only: bool = False) -> Dict[str, Any] | List[float]:
    """
    One full train/eval cycle.
    If return_history_only=True, we just return the PSO history
    (useful for heatmaps or multiple-run experiments).
    """
    # 1) load data
    x, y = load_data(config.DATASET_PATH)

    # 2) split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        train_ratio=config.TRAIN_RATIO,
        seed=config.RANDOM_SEED,
    )

    # 3) preprocess
    if config.USE_MINMAX:
        mins, maxs = fit_minmax(x_train)
        x_train_s = transform_minmax(x_train, mins, maxs)
        x_test_s = transform_minmax(x_test, mins, maxs)
    else:
        x_train_s = x_train
        x_test_s = x_test

    # 4) train (PSO + ANN)
    net, best_fit, history = train_with_pso(x_train_s, y_train)

    if return_history_only:
        return history  # just the PSO curve

    # 5) evaluate
    train_mae = evaluate_mae(net, x_train_s, y_train)
    test_mae = evaluate_mae(net, x_test_s, y_test)

    # 6) predictions (for charts)
    y_train_predictions = predict_all(net, x_train_s)
    y_test_predictions = predict_all(net, x_test_s)

    return {
        "net": net,
        "best_fit": best_fit,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_predictions,
        "y_test_pred": y_test_predictions,
        "history": history,
    }


def run_pipeline() -> None:
    result = run_single_training(return_history_only=False)

    net: Network = result["net"]
    best_fit: float = result["best_fit"]
    train_mae: float = result["train_mae"]
    test_mae: float = result["test_mae"]
    y_train = result["y_train"]
    y_test = result["y_test"]
    y_train_predictions = result["y_train_pred"]
    y_test_predictions = result["y_test_pred"]
    history = result["history"]

    # 7) report
    print("======== TASK 4 / TRAIN PIPELINE ========")
    print(f"Dataset:       {config.DATASET_PATH}")
    print(f"PSO best MAE:  {best_fit:.4f}")
    print(f"Train MAE:     {train_mae:.4f}")
    print(f"Test  MAE:     {test_mae:.4f}")

    # 8) charts
    # 8.1 PSO convergence
    if history:
        plots.plot_convergence(history)

    # 8.2 actual vs predicted
    plots.plot_actual_vs_predicted(y_train, y_train_predictions, title="Train: Actual vs Predicted")
    plots.plot_actual_vs_predicted(y_test, y_test_predictions, title="Test: Actual vs Predicted")

    # 8.3 error-style charts
    plots.plot_errors(y_test, y_test_predictions, title="Test: Absolute Errors")
    plots.plot_residuals(y_test, y_test_predictions, title="Test: Residuals")
    plots.plot_error_boxplot(y_train, y_train_predictions, y_test, y_test_predictions)
