# ann/__init__.py
"""
ANN Package
============

A lightweight, configurable Artificial Neural Network framework designed for
Particle Swarm Optimization (PSO)–based training, focused on regression tasks.

Modules:
--------
- activation : Activation functions (sigmoid, relu, tanh, linear)
- layer      : Dense layer implementation
- network    : Feedforward network of layers
- builder    : Utility to construct configurable networks
- loss       : Common loss functions (e.g., MSE)

Usage Example:
--------------
from ann import build_network

net = build_network([3, 5, 1], ['relu', 'linear'])
y_pred = net.forward([0.2, -0.5, 0.7])
print("Prediction:", y_pred)
"""

from .activation import ACTIVATIONS
from .layer import Dense
from .network import Network
from .builder import build_network
from .loss import mean_absolute_error

__all__ = [
    "ACTIVATIONS",
    "Dense",
    "Network",
    "build_network",
    "mean_absolute_error",
]

__version__ = "1.0.0"
