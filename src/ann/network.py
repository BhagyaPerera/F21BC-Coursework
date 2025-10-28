from typing import List, Sequence
from .layer import Dense

""" Feed forward neural network layers.Provides forward pass and flat parameter interface for PSO"""
""" Network is Chain of dense layers"""
class Network:

    #region constructor
    def __init__(self, layers: List[Dense]):
        if not layers:
            raise ValueError("Network must contain at least one layer")
        self.layers = layers
    #endregion

    #region number_parameters
    def number_parameters(self) -> int:
        return sum(l.number_parameters() for l in self.layers)
    #endregion

    #region forward algorithm
    def forward_algorithm(self, x: Sequence[float]) -> List[float]:
        output = list(x)
        for layer in self.layers:
            output = layer.forward_algorithm(output)
        return output
    #endregion

    #region setParameters
    def set_parameters(self, flat: List[float]) -> None:
        if len(flat) != self.number_parameters():
            raise ValueError(f"Expected {self.number_parameters()} parameters, got {len(flat)}")
        offset = 0
        for l in self.layers:
            offset = l.set_parameters(flat, offset)
    #endregion

    #region getParameters
    def get_parameters(self) -> List[float]:
        params: List[float] = []
        for l in self.layers:
            params.extend(l.gather_parameters())
        return params
    #endregion
