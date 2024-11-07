from neuralnetwork.layers.layer import Layer
import numpy as np
from typing import Optional
from neuralnetwork.optimizers.optimizer import Optimizer
from neuralnetwork.optimizers.sgd import SGD
from neuralnetwork.activations.activation import Activation
from neuralnetwork.activations.relu import ReLU
from neuralnetwork.neuron import Neuron 

class FullyConnectedLayer(Layer):
    def __init__(
        self,
        n_inputs: Optional[int] = None,
        n_neurons: Optional[int] = None,
        activation: Optional[Activation] = ReLU,
        optimizer: Optional[Optimizer] = SGD,
    ):
        super().__init__(n_inputs, n_neurons, activation, optimizer)
        
    def initialize_params(self, inputs):
        """
        Initializes Neurons with random weights and biases following the Normal Distr.
        """
        self.neurons = [Neuron(self.n_inputs) for _ in range(self.n_neurons)]
    
    def forward_propagation(self, inputs: np.ndarray) -> tuple[np.ndarray, int]:
        """
        inputs:
            An array of features (should be a numpy array)

        Returns:
            An array of the output values from each neuron in the layer.

        Function:
            Performs forward propagation by calculating the weighted sum for each neuron
        and applying the activation function
        """
        self.inputs = inputs
        activation_outputs = []

        for neuron in self.neurons:
            weighted_sum = neuron.calculate_neuron_weighted_sum(inputs)
            output = self.activation.apply(weighted_sum)
            activation_outputs.append(output)

        return np.array(activation_outputs)

    def back_propagation(self, y_orig, y_pred):
        # gradient wrt y_pred
        grads_and_vars = []

        for i, neuron in enumerate(self.neurons):
            dl_dz = self.calc_gradient_wrt_z(neuron.calculate_neuron_weighted_sum(self.inputs), y_pred, y_orig)

            # weights, bias
            dl_dw = self.calc_gradient_wrt_w(dl_dz, self.inputs)
            dl_db = self.calc_gradient_wrt_b(dl_dz)


            grads_and_vars.append((dl_dw, neuron.weights)) 
            grads_and_vars.append((dl_db, neuron.bias)) 

        # pass to optimizer
        grads_and_vars = self.optimizer.apply_gradients(grads_and_vars)

        idx = 0
        for neuron in self.neurons:
            neuron.weights = grads_and_vars[idx]
            idx += 1
            neuron.bias = grads_and_vars[idx]
            idx += 1