from ncxlib.neuralnetwork.layers import Layer
import numpy as np
from ncxlib.util import log


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

    def forward_propagation(self, inputs, no_save):
        return self.layer.forward_propagation(inputs, no_save)

    def back_propagation(self, y_true: np.ndarray, learning_rate: float) -> None:

        log(f"Backward Propagation for Layer {self.layer.name} based on True Label: {y_true}: -----")
    
        activated = np.clip(self.layer.activated, 1e-7, 1 - 1e-7)

        dl_da = (activated - y_true) / (activated * (1 - activated)) 

        da_dz = self.layer.activation.derivative(self.layer.z)

        dl_dz = dl_da * da_dz 

        log(f"activated_z: {activated}")
        log(f"dl_da: {dl_da}")
        log(f"da_dz: {da_dz}")
        log(f"dl_dz: {dl_dz}")
        log(f"self.layer.inputs.T: {self.layer.inputs.T}")
        log(f"self.layer.W: {self.layer.W}")

        self.layer.gradients = dl_dz
        self.layer.old_W = self.layer.W.copy()

        dw = dl_dz @ self.layer.inputs.T 
        log(f"dw: {dw}")
        self.layer.W -= learning_rate * dw

        # Calculate the bias gradient as the sum of dl_dz across all samples
        db = np.sum(dl_dz, axis=1, keepdims=True)  # Shape: (n_neurons, 1)
        self.layer.b -= learning_rate * db  # Update biases

        # Log the updated weights and biases if needed
        log(f"Updated Weights: {self.layer.W}")
        log(f"Updated Biases: {self.layer.b}")
