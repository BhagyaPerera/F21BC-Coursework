from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple,Dict,List

#region PSO_Configuration
@dataclass
class ANNConfig:
    """
    Configuration for building a feedforward ANN for the coursework.
    This lets you change the network shape without touching the training code.
    """

    # how many input features your dataset has
    input_dim: int = 8

    # how many outputs (for concrete = 1)
    output_dim: int = 1

    # list of hidden layers, in order
    # each item: {"units": <int>, "activation": <str from your activation.py>}
    hidden_layers: List[Dict[str, str | int]] = field(default_factory=lambda: [
        {"units": 16, "activation": "relu"},
        {"units": 8, "activation": "tanh"},
    ])

    # activation to use in the final/output layer
    # for regression this should be "identity"
    output_activation: str = "identity"
#endregion
