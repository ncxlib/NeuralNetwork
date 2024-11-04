from neuralnetwork.Layers import Layer
from neuralnetwork import Neuron
import numpy as np


class FullyConnectedLayer(Layer):
    def __init__(self, n_inputs: str, n_neurons: int, activation_fn: str):
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
