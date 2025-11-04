# tests/test_ann_manual.py (or just run in a notebook)
from src.ann.builder import build_network

# 2 inputs → 1 hidden layer (2 neurons, relu) → 1 output (identity)
net = build_network(
    input_dimension=2,
    layers_spec=[{"units": 2, "activation": "relu"}],
    output_dimension=1,
    output_activation="identity",
)

# we know how many params
print("num params:", net.number_parameters())  # should be small

# set params to something easy to check
# order must match your Dense.set_params(...)
# For example: W1(2x2) + b1(2) + W2(2x1) + b2(1)
params = [
    1.0, 0.0,      # W1 row 0
    0.0, 1.0,      # W1 row 1
    0.0, 0.0,      # b1
    1.0, 1.0,      # W2 (2 rows, 1 col)
    0.0, 0.0       # b2
]
net.set_parameters(params)

x = [3.0, 4.0]
y = net.forward_algorithm(x)
print("output:", y)
