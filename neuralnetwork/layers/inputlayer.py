from neuralnetwork.layers import Layer

class InputLayer(Layer):
    def __init__(self, layer: Layer, n_inputs = None, n_neurons = None, activation = ..., optimizer = ...):
        if layer:
            super().__init__(layer.n_inputs, layer.n_neurons, layer.activation, layer.optimizer)

    
    def forward_propagation(self, inputs):
        return inputs