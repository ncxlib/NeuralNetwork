from ncxlib.neuralnetwork.layers import Layer

class InputLayer(Layer):
    def __init__(
        self, n_inputs=None, n_neurons=None, activation=..., optimizer=...
    ):
        super().__init__(
            n_inputs, n_neurons, name="input"
        )
    
    def initialize_params(self, inputs):
        self.layer.initialize_params()

    def forward_propagation(self, inputs):
        return inputs
    
    def back_propagation(self, y_orig, y_pred):
        return super().back_propagation(y_orig, y_pred)
