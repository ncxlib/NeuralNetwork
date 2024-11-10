from typing import Optional
import numpy as np
from tqdm.auto import tqdm
from ncxlib.neuralnetwork.layers import Layer, InputLayer, OutputLayer
from ncxlib.neuralnetwork.losses import LossFunction, MeanSquaredError
from ncxlib.util import log

class NeuralNetwork:
    def __init__(self, layers: Optional[list[Layer]] = [], loss_fn: Optional[LossFunction] = MeanSquaredError):
        self.layers = layers
        self.compiled = False
        self.loss_fn = loss_fn

    def _compile(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.compiled = True

        try:
            targets = targets.astype(np.uint)
        except:
            raise ValueError("Labels should be of integer type, if they are categorical, please use OneHotEncoder Preprocessor")
            

        self.layers = [InputLayer(1, inputs.shape[1])] + self.layers

        previous_outputs = self.layers[0].n_neurons
        for layer in self.layers[1:]:
            if layer.n_inputs and layer.n_inputs != previous_outputs:
                raise ValueError(
                    "The inputs for a layer should match the number of neuron outputs of the previous layer."
                )

            if not layer.n_inputs:
                layer.n_inputs = previous_outputs

            previous_outputs = layer.n_neurons

        self.output_layer = OutputLayer(self.layers[-1], loss_fn = self.loss_fn)

        for layer in self.layers:
            log(layer)

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagate_all(self, input_vector):
        for layer in self.layers[1:]:
            input_vector = layer.forward_propagation(input_vector)
        return input_vector
    
    def forward_propagate_all_no_save(self, input_vector):
        for layer in self.layers[1:]:
            input_vector = layer.forward_propagation(input_vector, no_save=True)
        return input_vector

    def back_propagation(self, y_true, learning_rate) -> None:
        next_layer = self.output_layer.layer
        
        self.output_layer.back_propagation(
                    y_true, learning_rate
                )
        
        for layer in reversed(self.layers[1:-1]):
            layer.back_propagation(next_layer, learning_rate)
            next_layer = layer
        
    def train(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        epochs = 10,
        learning_rate=0.001,
    ):

        if not self.compiled:
            self._compile(inputs, targets)

        progress = tqdm(range(epochs))
        loss = np.inf

        for epoch in progress:
            progress.set_description(f"Epoch: {epoch} | Loss: {loss}")
            
            total_loss = 0
            
            for i in range(len(inputs)):

                input_vector = inputs[i]
                class_label = int(targets[i])
                
                y_true = np.zeros(self.layers[-1].n_neurons)
                y_true[class_label] = 1

                log(f"NEW DATA POINT: {i}| LABEL: {y_true}")

                output_activations = self.forward_propagate_all(input_vector)
                output_activations = np.clip(output_activations, 1e-7, 1 - 1e-7) 

                log(f"Size: true: {y_true.shape}, activations: {output_activations.shape}")

                sample_loss = self.loss_fn().compute_loss(y_true, output_activations)
                total_loss += sample_loss
                self.back_propagation(y_true, learning_rate)
                

            loss = total_loss / len(inputs)

    def predict(self, inputs):
        return [np.argmax(self.forward_propagate_all_no_save(input)) for input in inputs]

    def evaluate(self, inputs, targets):
        predictions = self.predict(inputs)
        accuracy = np.mean(predictions == targets)
        print(f"Accuracy: {accuracy * 100:.2f}%")

