import numpy as np
from logs import log


class Neuron:
    def __init__(self, inputs: np.ndarray):
        """
        Initializes Neuron class:
            1. Inputs: an array of feature inputs matching the n_inputs of a layer
            1. Weights - How much influence this input has on the overall outcome
            2. Bias - Added to the weighted sum of inputs. Helps shift activation Fn when weighted sum = 0.
        """
        self.inputs = inputs
        self.weights = np.random.randn(len(inputs)) * np.sqrt(2 / len(inputs))
        # self.weights = np.zeros(inputs.shape)
        self.bias = 0
        self.weighted_sum = 0

    def calculate_neuron_weighted_sum(self, no_save=False):
        """
        Params:

        Returns:
            The weighted sum as a float value.

        Functionality:
            Calculates the total weighted sum of all neuron inputs + the bias term,
            which can later be passed to an activation function. This is the pre-activation Fn.

            Weighted Sum: z = w1 ⋅ x1 + w2 ⋅ x2 + ...+ wn ⋅ xn + b
        """
        log(f"Calculating weighted sum: {self.weights.shape} * {self.inputs.shape} + {self.bias}")
        weighted_sum = np.dot(self.weights, self.inputs) + self.bias
        if not no_save:
            self.weighted_sum = weighted_sum 
        return weighted_sum
    
    def __repr__(self):
        return f"Neuron {{\n\tz: {self.weighted_sum}\n\tinputs: {self.inputs}\n\tweights: {self.weights}\n\tbias: {self.bias}\n}}"
