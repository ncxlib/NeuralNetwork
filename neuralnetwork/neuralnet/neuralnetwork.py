from neuralnetwork.layers.fullyconnectedayer import FullyConnectedLayer
from neuralnetwork.losses.losses import MSE
from typing import Callable
import numpy as np
from tqdm.auto import tqdm


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, input_vector):
        for layer in self.layers:
            input_vector, y_pred = layer.forward_propagation(input_vector)
        return y_pred

    def back_propagation(self, y_true, y_pred):
        for layer in reversed(self.layers):
            layer.back_propagation(y_true, y_pred)
    

    def train(self, inputs, targets, epochs, loss_fn : Callable):
        for epoch in tqdm(range(epochs), leave=True):
            preds = [0] * len(inputs)

            for i in range(len(inputs)):
                # iterate over every individual sample && its target label:
                input_vector = inputs[i]
                y_true = targets[i]

                y_pred = self.forward_propagation(input_vector)
                self.back_propagation(y_true, y_pred)

                preds[i] = y_pred
                print(preds[i], y_true)

            average_loss = loss_fn(preds, targets)

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")


    def predict(self, inputs):
        return [self.forward_propagation(i) for i in inputs]
    
    def accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_pred == y_true))
