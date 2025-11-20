from pathlib import Path

from src.ann import ANNConfig
from src.pso import PSOConfig
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


def run_single_training(ann_config,pso_config,return_history_only: bool = False) -> Dict[str, Any] | List[float]:
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
        seed=None,
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
    net, best_fit, history = train_with_pso(x_train_s,y_train,ann_config,pso_config)

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

def run_pipeline(
    ann_config: ANNConfig,
    pso_config: PSOConfig,
    runs: int = 5,
    callback=None
) -> dict:
    """
    Run PSO-ANN multiple times and:
      • Support GUI callback(iteration, best, run_index)
      • Return full run results including predictions
      • Keep your existing MATPLOTLIB visualizations
    """
    import numpy as np
    import matplotlib.pyplot as plt

    results = []
    histories = []
    train_maes = []
    test_maes = []
    gbest_vals = []

    print(f"\n======== EXPERIMENTAL PIPELINE ({runs} RUNS) ========")

    # -----------------------------
    # RUNS LOOP
    # -----------------------------
    for run_index in range(runs):
        print(f"\n Run {run_index+1}/{runs}")

        # Train one run
        single_result = run_single_training(
            ann_config=ann_config,
            pso_config=pso_config,
            return_history_only=False
        )

        # Extract PSO history (list of gBest per iteration)
        history = single_result["history"]

        # NEW: Trigger GUI callback for each iteration
        if callback is not None:
            for iteration, best in enumerate(history):
                callback(iteration, best, run_index)

        # Collect run-level results
        results.append(single_result)
        histories.append(history)
        train_maes.append(float(single_result["train_mae"]))
        test_maes.append(float(single_result["test_mae"]))
        gbest_vals.append(float(single_result["best_fit"]))

        print(
            f"   gBest={single_result['best_fit']:.4f} | "
            f"Train MAE={single_result['train_mae']:.4f} | "
            f"Test MAE={single_result['test_mae']:.4f}"
        )

    # -----------------------------
    # SUMMARY STATISTICS
    # -----------------------------
    mean_train = np.mean(train_maes)
    std_train = np.std(train_maes)
    mean_test = np.mean(test_maes)
    std_test = np.std(test_maes)
    mean_gbest = np.mean(gbest_vals)
    std_gbest = np.std(gbest_vals)

    print("\n======== EXPERIMENT SUMMARY (Mean ± SD) ========")
    print(f"Number of Runs:   {runs}")
    print(f"ANN Architecture: {' → '.join([f'{l['units']}({l['activation']})' for l in ann_config.hidden_layers])}")
    print(f"PSO Config:       swarm={pso_config.swarm_size}, iter={pso_config.iterations}, "
          f"αlpha={pso_config.alpha}, beta={pso_config.beta}, gamma={pso_config.gamma}, bounds={pso_config.bounds}")
    print(f"Average gBest:    {mean_gbest:.4f} ± {std_gbest:.4f}")
    print(f"Average Train MAE:{mean_train:.4f} ± {std_train:.4f}")
    print(f"Average Test MAE: {mean_test:.4f} ± {std_test:.4f}")
    print("===================================================")

    # -----------------------------
    # KEEPING YOUR EXISTING MATPLOTLIB DISPLAY
    # -----------------------------

    n_cols = runs
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 14))
    fig.subplots_adjust(hspace=1.2, top=0.9, bottom=0.05)

    if runs == 1:
        axes = np.expand_dims(axes, axis=1)

    row_titles = [
        "PSO Convergence",
        "Training Actual vs Predicted",
        "Test Actual vs Predicted"
    ]
    title_y_positions = [0.96, 0.62, 0.28]

    for row_idx, title in enumerate(row_titles):
        fig.text(
            0.5, title_y_positions[row_idx], title,
            ha="center", va="center",
            fontsize=10, fontweight="bold"
        )

    return {
        "results": results,
        "histories": histories,
        "train_maes": train_maes,
        "test_maes": test_maes,
        "gbest_vals": gbest_vals
    }

