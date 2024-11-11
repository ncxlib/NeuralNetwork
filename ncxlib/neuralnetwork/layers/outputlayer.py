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

        activated = np.clip(self.layer.activated, 1e-7, 1 - 1e-7)

        dl_da = (activated - y_true) / (activated * (1 - activated)) 

        da_dz = self.layer.activation.derivative(self.layer.z)

        dl_dz = dl_da * da_dz 

        self.layer.gradients = dl_dz
        self.layer.old_W = self.layer.W.copy()

        dw = dl_dz @ self.layer.inputs.T 
        self.layer.W -= learning_rate * dw

        db = np.sum(dl_dz, axis=1, keepdims=True)
        self.layer.b -= learning_rate * db