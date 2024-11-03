import numpy as np
import Layer
from NeuralNetwork import Neuron

class FullyConnectedLayer(Layer):
    def __init__(self, n_inputs, n_neurons, activation_fn=None):
        super().__init__(activation_fn)
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.initialize_params()

    def initialize_params(self):
        self.neurons = [Neuron(self.n_inputs) for _ in range(self.n_neurons)]

    # keep track of weighted sum of all neurons
    def forward_propogation(self):
        pass

    # apply activation fn
