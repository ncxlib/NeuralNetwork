import numpy as np

class Neuron:
    def __init__(self, n_inputs):
        '''
        Initializes Neuron class:
            1. Weights - How much influence this input has on the overall outcome
            2. Bias - Added to the weighted sum of inputs. Helps shift activation Fn when weighted sum = 0.
            3. set activation Fn
        '''
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()

    def calculate_neurons_weighted_sum(self, inputs):
        '''
        Calculates the total weighted sum of all neuron inputs + the bias term
        to later be passed to an activation Fn.

        Weighted Sum: z = w1⋅x_1+w_2⋅x_2+ ...+ w_n⋅x_n + b
        x_i = each input
        w_i = each weight
        '''
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        return weighted_sum
