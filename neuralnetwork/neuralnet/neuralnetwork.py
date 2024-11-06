from neuralnetwork.layers.fullyconnectedayer import FullyConnectedLayer
from neuralnetwork.losses.losses import MSE
from typing import Callable
import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, input_vector):
        for layer in self.layers:
            input_vector = layer.forward_propagation(input_vector)
        return input_vector

    def back_propagation(self, y_true, y_pred):
        for layer in reversed(self.layers):
            layer.back_propagation(y_true, y_pred)
    

    def train(self, inputs, targets, epochs, loss_fn : Callable):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(inputs)):
                # iterate over every individual sample && its target label:
                input_vector = inputs[i]
                y_true = targets[i]

                y_pred = self.forward_propagation(input_vector)

                loss = loss_fn(np.array([y_pred]), np.array([y_true]))
                total_loss += loss

                self.back_propagation(y_true, y_pred)


            average_loss = total_loss / len(inputs)
            print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

    def predict(self, inputs):
        return self.forward_propagation(inputs)
