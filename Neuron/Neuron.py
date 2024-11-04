import numpy as np

class Neuron:
    def __init__(self, n_inputs):
        """
        Initializes Neuron class:
            1. Weights - How much influence this input has on the overall outcome
            2. Bias - Added to the weighted sum of inputs. Helps shift activation Fn when weighted sum = 0.
        """
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()
        self.weighted_sum = 0

    def calculate_neuron_weighted_sum(self, inputs):
        """
        Params:
            inputs - an array of features (should be an np array for dot prod)

        Returns:
            The weighted sum as a float value.

        Functionality:
            Calculates the total weighted sum of all neuron inputs + the bias term,
            which can later be passed to an activation function. This is the pre-activation Fn.

            Weighted Sum: z = w1 ⋅ x1 + w2 ⋅ x2 + ...+ wn ⋅ xn + b
        """
        self.weighted_sum = np.dot(self.weights, inputs) + self.bias
        return self.weighted_sum