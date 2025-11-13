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

def run_pipeline(ann_config: ANNConfig, pso_config: PSOConfig, runs: int = 5) -> None:
    """
    Run PSO-ANN multiple times (default=5) and produce:
      • Per-run PSO convergence, train/test plots
      • Averaged PSO convergence and predicted-vs-actual plots (if runs > 1)
      • Prints mean ± SD summary
    """
    import numpy as np
    import matplotlib.pyplot as plt

    results, histories = [], []
    train_maes, test_maes, gbest_vals = [], [], []

    print(f"\n======== EXPERIMENTAL PIPELINE ({runs} RUNS) ========")

    # ---- Run experiments ----
    for i in range(runs):
        print(f"\n Run {i+1}/{runs}")
        result = run_single_training(ann_config, pso_config, return_history_only=False)
        results.append(result)
        histories.append(result["history"])
        train_maes.append(float(result["train_mae"]))
        test_maes.append(float(result["test_mae"]))
        gbest_vals.append(float(result["best_fit"]))
        print(f"   gBest={result['best_fit']:.4f} | Train MAE={result['train_mae']:.4f} | Test MAE={result['test_mae']:.4f}")

    # ---- Compute summary statistics ----
    mean_train, std_train = np.mean(train_maes), np.std(train_maes)
    mean_test, std_test = np.mean(test_maes), np.std(test_maes)
    mean_gbest, std_gbest = np.mean(gbest_vals), np.std(gbest_vals)

    # ---- Print concise summary ----
    print("\n======== EXPERIMENT SUMMARY (Mean ± SD) ========")
    print(f"Number of Runs:   {runs}")
    print(f"ANN Architecture: {' → '.join([f'{l['units']}({l['activation']})' for l in ann_config.hidden_layers])}")
    print(f"PSO Config:       swarm={pso_config.swarm_size}, iter={pso_config.iterations}, "
          f"αlpha={pso_config.alpha}, beta={pso_config.beta}, gamma={pso_config.gamma}, bounds={pso_config.bounds}")
    print(f"Average gBest:    {mean_gbest:.4f} ± {std_gbest:.4f}")
    print(f"Average Train MAE:{mean_train:.4f} ± {std_train:.4f}")
    print(f"Average Test MAE: {mean_test:.4f} ± {std_test:.4f}")
    print("===================================================")

    # ---- PLOTTING (3-row layout with spacing for individual runs) ----
    n_cols = runs
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 14))
    fig.subplots_adjust(hspace=1.2, top=0.9, bottom=0.05)

    # Handle the case when n_cols == 1 (single run)
    if runs == 1:
        axes = np.expand_dims(axes, axis=1)  # make it 2D for consistent indexing

    # Row titles
    row_titles = ["PSO Convergence", "Training Actual vs Predicted", "Test Actual vs Predicted"]
    title_y_positions = [0.96, 0.62, 0.28]
    for row_idx, title in enumerate(row_titles):
        fig.text(0.5, title_y_positions[row_idx], title,
                 ha='center', va='center', fontsize=10, fontweight='bold')

    # ---- Per-run plots ----
    for i, res in enumerate(results):
        # Row 1: PSO convergence
        axes[0, i].plot(histories[i], color="blue")
        axes[0, i].set_title(f"Run {i+1}", fontsize=10)
        axes[0, i].set_xlabel("Iteration")
        axes[0, i].set_ylabel("gBest")
        axes[0, i].grid(True)

        # Row 2: Train Actual vs Predicted
        axes[1, i].scatter(res["y_train"], res["y_train_pred"], color="green", s=10, alpha=0.6)
        axes[1, i].plot([min(res["y_train"]), max(res["y_train"])],
                        [min(res["y_train"]), max(res["y_train"])], 'r--')
        axes[1, i].set_title(f"Train MAE={res['train_mae']:.2f}", fontsize=10)
        axes[1, i].set_xlabel("Actual")
        axes[1, i].set_ylabel("Predicted")
        axes[1, i].grid(True)

        # Row 3: Test Actual vs Predicted
        axes[2, i].scatter(res["y_test"], res["y_test_pred"], color="purple", s=10, alpha=0.6)
        axes[2, i].plot([min(res["y_test"]), max(res["y_test"])],
                        [min(res["y_test"]), max(res["y_test"])], 'r--')
        axes[2, i].set_title(f"Test MAE={res['test_mae']:.2f}", fontsize=10)
        axes[2, i].set_xlabel("Actual")
        axes[2, i].set_ylabel("Predicted")
        axes[2, i].grid(True)

    fig.suptitle(f"ANN + PSO Experimental Results ({runs} Runs)", fontsize=6, y=0.995)
    plt.tight_layout(rect=[1, 0, 1, 0.94])
    plt.show()

    # ---- AVERAGE PLOTS (only if multiple runs) ----
    if runs > 1:
        print("\nGenerating averaged plots across runs...")

        # 1️.Average PSO Convergence
        max_len = max(len(h) for h in histories)
        padded_histories = np.array([np.pad(h, (0, max_len - len(h)), mode='edge') for h in histories])
        mean_curve = np.mean(padded_histories, axis=0)
        std_curve = np.std(padded_histories, axis=0)

        plt.figure(figsize=(8, 5))
        plt.plot(mean_curve, color='blue', label='Mean gBest')
        plt.fill_between(range(max_len),
                         mean_curve - std_curve,
                         mean_curve + std_curve,
                         color='blue', alpha=0.2)
        plt.title("Average PSO Convergence Across Runs")
        plt.xlabel("Iteration")
        plt.ylabel("gBest")
        plt.legend()
        plt.grid(True)
        plt.show()

        # 2.Average Train/Test Actual vs Predicted
        all_train_actual = np.concatenate([r["y_train"] for r in results])
        all_train_pred = np.concatenate([r["y_train_pred"] for r in results])
        all_test_actual = np.concatenate([r["y_test"] for r in results])
        all_test_pred = np.concatenate([r["y_test_pred"] for r in results])

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        # Train
        ax[0].scatter(all_train_actual, all_train_pred, color="green", s=10, alpha=0.5)
        ax[0].plot([min(all_train_actual), max(all_train_actual)],
                   [min(all_train_actual), max(all_train_actual)], 'r--')
        ax[0].set_title(f"Average Training (Mean MAE={mean_train:.2f})")
        ax[0].set_xlabel("Actual")
        ax[0].set_ylabel("Predicted")
        ax[0].grid(True)
        # Test
        ax[1].scatter(all_test_actual, all_test_pred, color="purple", s=10, alpha=0.5)
        ax[1].plot([min(all_test_actual), max(all_test_actual)],
                   [min(all_test_actual), max(all_test_actual)], 'r--')
        ax[1].set_title(f"Average Testing (Mean MAE={mean_test:.2f})")
        ax[1].set_xlabel("Actual")
        ax[1].set_ylabel("Predicted")
        ax[1].grid(True)
        plt.tight_layout()
        plt.show()

    print("visualization completed.")








