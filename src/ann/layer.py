from typing import List, Sequence, Callable
from .activation import ACTIVATIONS

"""Define Dense (fully connected layer) for feed forward neural network.

     1.It Takes a set of inputs (numbers).-> x
     2.Multiple by Weights.-> W
     3.Add Biases.-> b
     4.Pass the results through activation function.->f
     
     output=f(W.x+b)
"""

#region class Dense
class Dense:

    #region constructor
    def __init__(self, input_dimension: int, output_dimension: int, activation_function: str = 'relu') -> None:
        if activation_function not in ACTIVATIONS:
            raise KeyError(f"Invalid activation function: {activation_function}")
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.activation_function = activation_function
        self.activation: Callable[[float], float] = ACTIVATIONS[activation_function]

        self.weights:List[List[float]]=[[0.0]*input_dimension for i in range(output_dimension)]
        self.bias:List[float]=[0.0]*output_dimension
    #endregion

    """ this method returns total number of weights and biases """
    #region get numberOf_params
    def number_parameters(self) -> int:
        return self.input_dimension * self.output_dimension + self.input_dimension
    #endregion

    """long 1-D list flat that contains all parameters for this layer (weights first, then biases),and an offset telling where to start reading from in that list."""
    #region set_parameters
    def set_parameters(self, flat: List[float], offset: int) -> int:
        """Unpack weights/biases from a flat vector starting at offset."""

        count_weights = self.output_dimension * self.input_dimension   #weight vector dimension

        for i in range(self.output_dimension):
            start = offset + i * self.input_dimension                  #start index of weight array
            end = start + self.input_dimension                         #ending index of weight array
            self.weights[i] = flat[start:end]                          #read weights and add to weights array

        #offset for bias
        offset += count_weights                                        #make offset for bias

        self.bias = flat[offset: offset + self.output_dimension]       #read bias and add to bias array
        offset += self.output_dimension                                #change offset
        return offset
    #endregion

    """This method packs the layer’s parameters into one long 1-D list"""
    #region get_parameters
    def gather_parameters(self) -> List[float]:
        flat: List[float] = []                                        #initiate 1D empty list flat
        for row in self.weights:                                      #add weights to flat list
            flat.extend(row)
        flat.extend(self.bias)                                        #add bias to flat list
        return flat
    #endregion

    """This method is to use forward algorithm using weights, inputs and biases"""
    #region forward_algorithm
    def forward_algorithm(self, input_x: Sequence[float]) -> List[float]:
        if len(input_x) != self.input_dimension:                            #check the input vector length is equal to input neuron dimension.
            raise ValueError(f"Expected {self.input_dimension} inputs, got {len(input_x)}")           #if not raise an error
        output_y: List[float] = []                                                                    #create an empty list output
        for i in range(self.output_dimension):
            z = sum(weight*xj for weight, xj in zip(self.weights[i], input_x)) + self.bias[i]
            output_y.append(self.activation(z))                                                      #calculate output=f(w.x+b) for each input and append to output_y list.
        return output_y
    #endregion

#endregion
