from neuralnetwork.layers.layer import Layer
import numpy as np
from typing import Optional
from logs import log
from neuralnetwork.optimizers.optimizer import Optimizer
from neuralnetwork.optimizers.sgd import SGD
from neuralnetwork.activations.activation import Activation
from neuralnetwork.activations.relu import ReLU
from neuralnetwork.neuron import Neuron
from neuralnetwork.losses import LossFunction, MeanSquaredError

class FullyConnectedLayer(Layer):
    def __init__(
        self,
        n_inputs: Optional[int] = None,
        n_neurons: Optional[int] = None,
        activation: Optional[Activation] = ReLU,
        optimizer: Optional[Optimizer] = SGD,
        loss_fn: Optional[LossFunction] = MeanSquaredError,
        name: Optional[str] = ""
    ):
        super().__init__(n_inputs, n_neurons, activation, optimizer, loss_fn, name=name)

    def initialize_params(self, inputs):
        """
        Initializes Neurons with random weights and biases following the Normal Distr.
        """

        if not self.neurons:
            log(f"Initializing Neurons for Layer {self.name}")
            self.neurons = [Neuron(inputs) for _ in range(self.n_neurons)]
        else:
            log(f"Updating Neuron Inputs for Layer {self.name}")
            for neuron in self.neurons:
                neuron.inputs = inputs

    def forward_propagation(self, inputs: np.ndarray, no_save: Optional[bool] = False) -> tuple[np.ndarray, int]:
        """
        inputs:
            An array of features (should be a numpy array)

        Returns:
            An array of the output values from each neuron in the layer.

        Function:
            Performs forward propagation by calculating the weighted sum for each neuron
        and applying the activation function
        """
        self.initialize_params(inputs)
        activation_outputs = []

        log(f"Forward Propogation for Layer {self.name} -----")

        for neuron in self.neurons:
            weighted_sum = neuron.calculate_neuron_weighted_sum(no_save)
            log(neuron)
            output = self.activation.apply(weighted_sum)
            activation_outputs.append(output)

        log(f"Results: {activation_outputs}")
        return np.array(activation_outputs)
    
    def back_propagation(self, next_layer: Layer, learning_rate: float) -> np.ndarray:
        
        log(f"Backward Propogation for Layer {self.name} based on {next_layer.name}: -----")
        for index, neuron in enumerate(self.neurons):
            
            log(f"\tNeuron: {neuron}")

            da_dz = self.activation.derivative(neuron.weighted_sum)

            dl_dz = 0
            for next_neuron in next_layer.neurons:
                dl_dznext_i = next_neuron.gradient
                dznext_da = next_neuron.old_weights[index]
                dl_dz += dl_dznext_i * dznext_da * da_dz

            neuron.old_weights = neuron.weights.copy()

            log(f"\tda_dz = {da_dz}")
            log(f"\tdl_dz = {dl_dz}")

            
            # TODO: Switch to optimizer
            for i in range(len(neuron.weights)):
                dz_dwi = neuron.inputs[i]
                dl_dwi = dl_dz * dz_dwi
                neuron.weights[i] -=  learning_rate * dl_dwi

            neuron.bias -= learning_rate * dl_dz

            log(f"\tUpdated Weights: {neuron.weights}")
            log(f"\tUpdated Bias: {neuron.bias}")

            neuron.gradient = dl_dz