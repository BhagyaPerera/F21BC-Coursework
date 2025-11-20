"""
Sanity check for ANN + PSO coupling.

What it does:
1. builds a tiny fake dataset (8 features -> 1 target)
2. calls train_ann_with_pso(...)
3. prints best loss
4. prints predictions for the same data

Run from project root:
    python -m src.Test.ann_pso_test
"""

from src.ann_pso import make_ann_objective,train_ann_with_pso
from src.ann.builder import build_network


def main():
    # 1) tiny fake dataset (same shape as concrete dataset: 8 inputs)
    # here I'm making an easy pattern so PSO can learn it
    x_train = [
        [1, 1, 1, 1, 1, 1, 1, 1],   # sum = 8
        [2, 0, 0, 0, 0, 0, 0, 0],   # sum = 2
        [0, 0, 0, 0, 1, 1, 1, 1],   # sum = 4
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # sum = 4
    ]
    # target: just use the sum as "correct" output
    y_train = [8.0, 2.0, 4.0, 4.0]

    # 2) train ANN using PSO
    net, best_loss,history = train_ann_with_pso(x_train, y_train)

    print("=== ANN + PSO sanity check ===")
    print("Best loss (MAE) found by PSO:", best_loss)

    # 3) test the trained network on the same inputs
    for x_input, y_input in zip(x_train, y_train):
        predictions = net.forward_algorithm(x_input)[0]
        print(f"input={x_input}  target={y_input:.3f}  prediction={predictions:.3f}")


if __name__ == "__main__":
    main()
