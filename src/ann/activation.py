import math
from typing import Callable, Dict

""" Activation functions are what make a neural network non-linear.
It allows each neuron the ability to learn curves and complex patterns rather than just straight lines.
Here we are using following activation functions:
          1. Logistic
          2. ReLU
          3. Hyperbolic tangent
          4. Identity """

#region activation functions

#region logistic function
def logistic_function(x:float)->float:
                                                     #for large +ve, -ve x -->exp(-x) or exp(x) overflow or underflow.To avoid that we separate non-negative and negative
    if x >= 0:                                       #check for the non-negative inputs.
        z = math.exp(-x)                             #compute z for non-negative inputs. for large positive x z value is tiny.
        return 1 / (1 + z)                           #calcualte standard sigmoid. for large positive x it will close to 1. return value.

    else:
        z = math.exp(x)                              #check for the negative inputs. compute z.
        return z / (1 + z)                           #calculate sigmoid and return.
#endregion

#region relu function
def relu_function(x: float) -> float:
    return x if x > 0 else 0.0                       #relu function returns value greater than for equal 0.
#endregion

#region hyperbolic tangent function
def tanh_function(x: float) -> float:
    return math.tanh(x)                              #for large negative tan(x)-->-1,lagre positive tan(x)--> 1, 0-->tan(x)-->0
#endregion

#region Identity function
def identity_function(x: float) -> float:
    return x                                         #f(x)=x output is equal to input
#endregion

#region activations dictionary
ACTIVATIONS: Dict[str, Callable[[float], float]] = { #Registry of activation function. create dictionary of activations
    "logistic": logistic_function,
    "relu": relu_function,
    "tanh": tanh_function,
    "identity": identity_function,
}
#endregion

#endregion

