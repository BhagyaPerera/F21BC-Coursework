import random
from typing import List, Dict, Any
from .layer import Dense
from .network import Network

"""Helper functions to construct a configurable ANN and
generate random weight vectors for PSO initialisation."""

#region build_network
def build_network(input_dimension: int,layers_spec: List[Dict[str, Any]],output_dimension: int,output_activation: str = "identity") -> Network:

    layers: List[Dense] = []                                                                   #create layers with type Dense List.
    for spec in layers_spec:
        layers.append(Dense(input_dimension, int(spec["units"]), spec["activation"]))          #iterate through hidden layer blue-print.
        input_dimension = int(spec["units"])
    layers.append(Dense(input_dimension, output_dimension, output_activation))                 #add output layer
    return Network(layers)                                                                     #return Network
#endregion


"""Purpose: tiny wrapper that returns how many scalar parameters the whole Network has (all weights + all biases across all layers).
Why: PSO needs the dimension of the search space. That dimension is exactly network.num_params() because each particle’s position is a flat vector of all network parameters."""
#region parameter_count
def parameter_count(network: Network) -> int:
    return network.number_parameters()
#endregion


"""Defines a helper that returns a flat list of random numbers to fill all weights + biases of network.
  low, high set the sampling range (default [-0.5, 0.5]).
  seed makes results reproducible (same random params every run if pass a fixed integer)."""
#region random_parameters
def random_parameters(network: Network, low: float = -0.5, high: float = 0.5, seed: int | None = None) -> List[float]:
    rng = random.Random(seed)
    return [rng.uniform(low, high) for _ in range(network.number_parameters())]
#endregion
