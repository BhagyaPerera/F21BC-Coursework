# src/Sanity_Check/pso_test.py

"""
Sanity tests for PSO implementation.

We test two things:
1. 1D sphere  : f(x) = x^2
2. 5D sphere  : f(x) = sum(x_i^2)

If PSO is correct, both should give a fitness very close to 0.

Run this from the project root:
    python -m src.Sanity_Check.pso_test
"""

from src.pso import PSO, PSOConfig  # uses your __init__.py re-exports


# ---------------------------
# 1) objective functions
# ---------------------------
def sphere_1d(position):
    """position is a list like [x]; return x^2"""
    x = position[0]
    return x * x


def sphere_nd(position):
    """n-D sphere: minimum at all zeros"""
    return sum(v * v for v in position)


# ---------------------------
# 2) tests
# ---------------------------
def test_pso_1d():
    # config: small swarm, few iters = fast
    cfg = PSOConfig(
        swarm_size=20,
        iterations=60,
        bounds=(-5.0, 5.0),   # search inside [-5, 5]
        minimize=True,
        k_informants=3,
    )

    # PSO expects (dimension, fitness_fn, config)
    pso = PSO(dimension=1, fitness_fn=sphere_1d, config=cfg)

    best_pos, best_val,history = pso.run(verbose=False)

    print("=== 1D test ===")
    print("best_pos:", best_pos)
    print("best_val:", best_val)   # should be near 0


def test_pso_nd(dim: int = 5):
    cfg = PSOConfig(
        swarm_size=30,
        iterations=100,
        bounds=(-10.0, 10.0),
        minimize=True,
        k_informants=3,
    )

    pso = PSO(dimension=dim, fitness_fn=sphere_nd, config=cfg)

    best_pos, best_val,history = pso.run(verbose=False)

    print(f"=== {dim}D test ===")
    print("best_pos:", best_pos)
    print("best_val:", best_val)   # should be near 0


if __name__ == "__main__":
    test_pso_1d()
    test_pso_nd(5)
