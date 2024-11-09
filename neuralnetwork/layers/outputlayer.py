from neuralnetwork.layers import Layer
import numpy as np
from logs import log


class OutputLayer(Layer):
    def __init__(
        self,  layer: Layer, loss_fn, n_inputs=None, n_neurons=None, activation=..., optimizer=...
    ):
        if layer:
            self.layer = layer
            layer.loss_fn = loss_fn
            super().__init__(
                layer.n_inputs, layer.n_neurons, layer.activation, layer.optimizer, loss_fn=loss_fn
            )

    def initialize_params(self, inputs):
        self.layer.initialize_params(inputs)

    def calculate_loss(self, y_pred: np.ndarray, y_orig: np.ndarray):
        return self.loss_fn.calculate_loss(y_pred, y_orig)

    def forward_propagation(self, inputs, no_save=False):
        return super().forward_propagation(inputs, no_save)

    def back_propagation(self, y_true: np.ndarray, learning_rate: float) -> None:

        log(f"Backward Propogation for Layer {self.layer.name} based on True Label: {y_true}: -----")
        for index, neuron in enumerate(self.layer.neurons):
            
            log(f"\tNeuron: {neuron}")

            activated = self.activation.apply(neuron.weighted_sum)

            activated = np.clip(activated, 1e-7, 1 - 1e-7)

            log(f"\ta(z) = {activated}")
            log(f"\ty_true for neuron {index} = {y_true[index]}")

            # For cross entropy right now
            dl_da = (activated - y_true[index]) / (activated * (1 - activated))

            da_dz = self.activation.derivative(neuron.weighted_sum)

            dl_dz = dl_da * da_dz

            log(f"\tdl_da = {dl_da}")
            log(f"\tda_dz = {da_dz}")
            log(f"\tdl_dz = {dl_dz}")
            
            neuron.old_weights = neuron.weights.copy()

            # TODO: Switch to optimizer
            for i in range(len(neuron.weights)):
                dz_dwi = neuron.inputs[i]
                dl_dwi = dl_dz * dz_dwi
                neuron.weights[i] -=  learning_rate * dl_dwi

            neuron.bias -= learning_rate * dl_dz

            log(f"\tUpdated Weights: {neuron.weights}")
            log(f"\tUpdated Bias: {neuron.bias}")

            neuron.gradient = dl_dz
