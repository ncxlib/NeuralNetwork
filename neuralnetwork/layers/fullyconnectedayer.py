from neuralnetwork.layers.layer import Layer
from neuralnetwork.neuron.neuron import Neuron
import numpy as np
from typing import Callable


class FullyConnectedLayer(Layer):
    def __init__(self, n_inputs: int, n_neurons: int, activation_fn: Callable):
        super().__init__(activation_fn)
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.initialize_params()

    def initialize_params(self):
        """
        Initializes Neurons with random weights and biases following the Normal Distr.
        """
        self.neurons = [Neuron(self.n_inputs) for _ in range(self.n_neurons)]

    def forward_propagation(self, inputs):
        """
        inputs:
            An array of features (should be a numpy array)

        Returns:
            An array of the output values from each neuron in the layer.

        Function:
            Performs forward propagation by calculating the weighted sum for each neuron
        and applying the activation function
        """
        activation_outputs = []

        for neuron in self.neurons:
            weighted_sum = neuron.calculate_neuron_weighted_sum(inputs)
            output = self.activation_fn(weighted_sum)
            activation_outputs.append(output)

        return np.array(activation_outputs)
    
    def calculate_loss(self, pred_y : np.ndarray, y_orig : np.ndarray):
        '''
        Takes in the labels predicted and original labels.

        Calculates the loss as a mean sq error
        '''
        mse = np.mean((pred_y - y_orig) ** 2)
        return mse

    # (feature1, feature1 (gradient of feture))
    def stochastic_gradient_descent(self, weights, bias):

        # use chain rule to calculate gradients for weights && bias 
        pass

    # TODO: We assume a MSE loss. When we implement CE we can update to include CE_WRT_Y_PRED
    def calc_gradient_wrt_y_pred(self, y_pred, y_orig):
       '''
        Calculates the gradient of the MSE loss wrt y_pred

        Params:
            y_pred = predicted label y
            y_orig = original label y

        Returns:
            Gradient of the loss wrt y_pred
       '''
       return 2 * (y_pred - y_orig) / self.n_inputs

    def back_propagation(self):
        pass

    def update_params(self):
        pass
    
    # compile(optimizer, type_of_loss=MSE)
    # fit(features (x), labels(y), epochs, validation_set(x (validation set), y (validation set))) 
        # back propagation happens here
    # evaluate(test_x, test_y) --> outputs loss && accuracy

    # neural net class (model) takes in array of layers
    # each base layer takes in (n_neurons, activation_fn)
