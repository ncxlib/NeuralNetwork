from neuralnetwork.layers import Layer
import numpy as np 

class OutputLayer(Layer):
    def __init__(self, layer: Layer, n_inputs = None, n_neurons = None, activation = ..., optimizer = ...):
        if layer:
            super().__init__(layer.n_inputs, layer.n_neurons, layer.activation, layer.optimizer)

    def calculate_loss(self, y_pred: np.ndarray, y_orig: np.ndarray):
        return self.loss_fn(y_pred, y_orig)
    
    def forward_propagation(self, inputs):
        activations = super().forward_propagation(inputs)

        highest = np.argmax(activations) if self.n_neurons > 1 else (1 if activations[0] >= 0.5 else 0)
        return activations, highest
    
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
        
    