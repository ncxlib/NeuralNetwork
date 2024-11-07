from typing import Callable, Optional
import numpy as np
from tqdm.auto import tqdm
from neuralnetwork.layers import Layer, InputLayer, OutputLayer

class NeuralNetwork:
    def __init__(self, layers: Optional[list[Layer]] = []):
        self.layers = layers
        self.compiled = False

    def _compile(self, inputs: np.ndarray) -> None:
        self.compiled = True

        self.layers[0].n_inputs = inputs.shape[1]

        previous_outputs = self.layers[0].n_neurons
        for layer in self.layers[1:]:
            if layer.n_inputs and layer.n_inputs != previous_outputs:
                raise ValueError("The inputs for a layer should match the number of neuron outputs of the previous layer.")
        
            if not layer.n_inputs:
                layer.n_inputs = previous_outputs
            
            previous_outputs = layer.n_neurons


        self.input_layer = InputLayer(self.layers[0])
        self.output_layer = OutputLayer(self.layers[-1])


    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagate_all(self, input_vector):
        for layer in self.layers:
            outputs = layer.forward_propagation(input_vector)
            if layer == self.output_layer:
                input_vector, y_pred = outputs
            else: 
                input_vector = outputs
        return y_pred

    def back_propagation(self, y_pred):
        for layer in reversed(self.layers[:-1]):
            layer.back_propagation(y_pred)
    

    def train(self, inputs: np.ndarray, targets: np.ndarray, epochs, loss_fn : Callable):

        if not self.compiled:
            self._compile(inputs)

        for epoch in tqdm(range(epochs), leave=True):
            preds = [0] * len(inputs)

            for i in range(len(inputs)):
                
                # iterate over every individual sample && its target label:
                input_vector = inputs[i]
                y_orig = targets[i]

                y_pred = self.forward_propagate_all(input_vector)

                self.output_layer.back_propagation(y_orig, y_pred)
                self.back_propagation(y_orig, y_pred)

                preds[i] = y_pred
                print(preds[i], y_orig)

            average_loss = loss_fn(preds, targets)

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")


    def predict(self, inputs):
        return [self.forward_propagation(i) for i in inputs]
    
    def accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_pred == y_true))
