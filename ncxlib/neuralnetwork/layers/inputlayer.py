from ncxlib.neuralnetwork.layers import Layer

class InputLayer(Layer):
    def __init__(
        self, n_neurons=None, n_inputs=None,  activation=..., optimizer=...
    ):
        super().__init__(
            n_neurons, n_inputs, name="input"
        )
    
    def initialize_params(self, inputs):
        self.layer.initialize_params()

    def forward_propagation(self, inputs):
        return inputs
    
    def back_propagation(self, y_orig, y_pred):
        return super().back_propagation(y_orig, y_pred)
