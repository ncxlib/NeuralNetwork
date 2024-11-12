from typing import Optional
import numpy as np
from tqdm.auto import tqdm
from ncxlib.neuralnetwork.layers import Layer, InputLayer, OutputLayer
from ncxlib.neuralnetwork.losses import LossFunction, MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy
from ncxlib.neuralnetwork.activations import ReLU, Sigmoid, Softmax
from ncxlib.util import log, timer, show_time, time_this
from ncxlib.neuralnetwork.layers import FullyConnectedLayer
import h5py

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
        
    @timer
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
                
                y_true = np.zeros((self.layers[-1].n_neurons, 1))
                y_true[class_label] = 1

                output_activations = self.forward_propagate_all(input_vector)

                output_activations = np.clip(output_activations, 1e-7, 1 - 1e-7) 
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
    
    def save_model(self, filepath):
        '''
        Function: 
            Saves the model as a .h5 file that stores each layers neurons, weights, bias, loss fn and
            activation function.

            ** Note: Do not add .h5 to the end of your filepath. This will be added automatically.
        '''
        h5_suffix = ".h5"
        final_path = filepath + h5_suffix
        with h5py.File(final_path, 'w') as f:
            loss_fn_name = self.loss_fn.__name__ if hasattr(self.loss_fn, '__name__') else self.loss_fn.__class__.__name__
            f.attrs['loss_function'] = loss_fn_name
            f.attrs['num_layers'] = len(self.layers)

            for i, layer in enumerate(self.layers):
                if layer.W is not None and layer.b is not None:
                    f.create_dataset(f"layer_{i}_weights", data=layer.W)
                    f.create_dataset(f"layer_{i}_bias", data=layer.b)
                    f.attrs[f"layer_{i}_activation"] = layer.activation.__class__.__name__
                else:
                    print(f"Skipping layer {i} as it has no weights or biases")  

        print(f"Model saved to {final_path}")

    
    @classmethod
    def load_model(cls, filepath):
        
        # TODO: Get rid of these lookup maps and add a _registry in LossFunction
        loss_fn_lookup = {
            "BinaryCrossEntropy": BinaryCrossEntropy,
            "MeanSquaredError": MeanSquaredError,
        }

        activation_fn_lookup = {
            "Sigmoid": Sigmoid,
            "ReLU": ReLU,
            "Softmax": Softmax
        }

        with h5py.File(filepath, 'r') as f:
            loss_fn_name = f.attrs['loss_function']
            loss_fn_class = loss_fn_lookup.get(loss_fn_name)

            if loss_fn_class is None:
                raise ValueError(f"loss fn {loss_fn_name} not found")
            
            model = cls(loss_fn=loss_fn_class())
            num_layers = f.attrs['num_layers']

            for i in range(1, num_layers):
                activation_fn_name = f.attrs.get(f"layer_{i}_activation")
                activation_fn_class = activation_fn_lookup.get(activation_fn_name)

                if activation_fn_class is None:
                    raise ValueError(f"Activation function '{activation_fn_name}' not found in lookup dictionary")
                
                weights = f[f"layer_{i}_weights"][:]
                biases = f[f"layer_{i}_bias"][:]
                n_neurons, n_inputs = weights.shape

                layer = FullyConnectedLayer(
                    n_inputs=n_inputs,
                    n_neurons=n_neurons,
                    activation=activation_fn_class()
                )
                layer.W = weights
                layer.b = biases
                model.layers.append(layer)
            
            print(f"model loaded from {filepath}")
            return model

    
    # verify final wts/bias against saved models wts/bias
    def print_final_weights_biases(self):
        print("Final Weights and Biases After Training:")
        for i, layer in enumerate(self.layers):
            if layer.W is not None and layer.b is not None:
                print(f"Layer {i}:")
                print(f"  Weights:\n{layer.W}")
                print(f"  Bias:\n{layer.b}")
            else:
                print(f"Layer {i} has no weights or biases")



   