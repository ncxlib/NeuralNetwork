from typing import Optional
import numpy as np
from tqdm import tqdm
from ncxlib.neuralnetwork.layers import Layer, InputLayer, OutputLayer
from ncxlib.neuralnetwork.losses import LossFunction, MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy
from ncxlib.neuralnetwork.activations import ReLU, Sigmoid, Softmax, LeakyReLU, Tanh
from ncxlib.util import log, timer, show_time, time_this
from ncxlib.neuralnetwork.layers import FullyConnectedLayer
import h5py, uuid

class NeuralNetwork:
    def __init__(self, layers: Optional[list[Layer]] = [], loss_fn: Optional[LossFunction] = MeanSquaredError):
        
        self.layers = layers
        self.compiled = False
        self.loss_fn = loss_fn

    def _compile(self, inputs: np.ndarray, targets: np.ndarray, learning_rate: float) -> None:
        self.compiled = True

        try:
            targets = targets.astype(np.uint)
        except:
            raise ValueError("Labels should be of integer type, if they are categorical, please use OneHotEncoder Preprocessor")
            

        self.layers = [InputLayer(n_neurons=inputs.shape[1], n_inputs=1)] + self.layers

        previous_outputs = self.layers[0].n_neurons
        for layer in self.layers[1:]:
            if layer.n_inputs and layer.n_inputs != previous_outputs:
                raise ValueError(
                    "The inputs for a layer should match the number of neuron outputs of the previous layer."
                )
                
            if not layer.n_inputs:
                layer.n_inputs = previous_outputs

            previous_outputs = layer.n_neurons

            layer.activation = layer.activation()

        self.output_layer = OutputLayer(layer=self.layers[-1], loss_fn = self.loss_fn)
        self.layers[-1].activation = self.output_layer.activation


    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagate_all(self, inputs_vector):
        for layer in self.layers[1:]:
            inputs_vector = layer.forward_propagation(inputs_vector)
        return inputs_vector
    
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
        epochs=10,
        learning_rate=0.001,
        batch_size=32,
        shuffle=True,
    ):
        if not self.compiled:
            self._compile(inputs, targets, learning_rate)

        progress = tqdm(range(epochs))
        loss = np.inf

        num_classes = len(np.unique(targets))

        for epoch in progress:
            progress.set_description(f"Epoch: {epoch} | Loss: {loss}")
            
            total_loss = 0

            if shuffle:
                indices = np.arange(len(inputs))
                np.random.shuffle(indices)
                inputs = inputs[indices]
                targets = targets[indices]

            for i in range(0, len(inputs), batch_size):

                X_batch = inputs[i:i + batch_size]
                y_batch = targets[i:i + batch_size]
                y_true = y_batch.reshape((len(y_batch), 1))

                if self.layers[-1].n_neurons > 1:
                    y_true = np.zeros((len(y_batch), self.layers[-1].n_neurons))
                    for j, class_label in enumerate(y_batch):
                        y_true[j, int(class_label)] = 1

                output_activations = self.forward_propagate_all(X_batch)

                # print(f"y_true: {y_true}, y_pred: {output_activations}")
                batch_loss = self.loss_fn().compute_loss(y_true, output_activations)
                total_loss += batch_loss

                self.back_propagation(y_true, learning_rate)

            # Compute average loss for the epoch
            loss = total_loss / len(inputs)
        

    def predict(self, inputs: np.ndarray, multiple=True, raw=False):
        """
        Predicts outputs for the given inputs.

        Parameters:
            inputs (np.ndarray): Input data.
            multiple (bool): If True, returns predictions for each input individually.
            raw (bool): If True, returns raw probabilities/logits instead of class predictions.
            threshold (float): Threshold for binary classification when using probabilities.

        Returns:
            np.ndarray or List: Predicted class labels or raw probabilities/logits.
        """
        activations = [self.forward_propagate_all_no_save(input) for input in inputs]

        # binary classification
        if self.layers[-1].n_neurons == 1:

            # assumes -1, 1 labels
            if min(activations) < 0:
                predictions = [1 if p >= 0 else -1 for p in activations]
            
            # assumes 0, 1 labels
            else:
                predictions = [1 if p >= 0.5 else 0 for p in activations]

        else: 
            predictions = [np.argmax(p) for p in activations]

        
        probabilities = np.array([a[0][0] if self.layers[-1].n_neurons == 1 else a[0] for a in activations])
        return (predictions, probabilities) if multiple else (predictions[0], probabilities[0])

    def evaluate(self, inputs, targets, metrics=["classification"], show=True):
        """
        Evaluates the model on given inputs and targets.

        Parameters:
            inputs (np.ndarray): Input data.
            targets (np.ndarray): True class labels.

        Returns:
            float: Accuracy of the model on the given data.
        """ 

        unique_targets = np.unique(targets)
        positive_class, negative_class = 1, 0

        # for +ve / -ve classes
        if np.sum(unique_targets >= 0) != len(unique_targets):
            positive_class = unique_targets[unique_targets > 0][0]
            negative_class = unique_targets[unique_targets < 0][0]
        
        results = {}

        predictions, probabilities = self.predict(inputs, multiple=True)

        for metric in metrics:
            if metric == "classification":
                accuracy = np.mean(np.array(predictions) == np.array(targets))
            
            elif metric == "balanced":
                
                TP = np.sum((targets == positive_class) & (predictions == positive_class))
                TN = np.sum((targets == negative_class) & (predictions == negative_class))
                FP = np.sum((targets == negative_class) & (predictions == positive_class))
                FN = np.sum((targets == positive_class) & (predictions == negative_class))

                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                accuracy = (sensitivity + specificity) / 2

            elif metric == "roc":
                accuracy = self.calculate_roc_area(probabilities, targets, positive_class)
                
            results[metric] = accuracy

        if show: print(results)
        return results
                

    def calculate_roc_area(self, probabilities, targets, positive_class):
        
        unique_vals = np.unique(probabilities)
        if set(unique_vals) == {-1, 1}:
            probabilities = (probabilities + 1) / 2
            
        probabilities = (probabilities + 1) / 2 

        targets = targets == positive_class

        sorted_indices = np.argsort(probabilities, kind="mergesort")[::-1]
        probabilities = probabilities[sorted_indices]
        targets = targets[sorted_indices]

        weight = 1.0

        distinct_indices = np.where(np.diff(probabilities))[0]
        threshold_idxs = np.r_[distinct_indices, len(targets) - 1]

        arr = (1 - targets) * weight
        out = np.cumsum(arr, axis=None, dtype=np.float64)

        tps = out[threshold_idxs]

        fps = 1 + threshold_idxs - tps
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]

        thresholds = probabilities[threshold_idxs]
        thresholds = np.r_[np.inf, thresholds]

        if fps[-1] <= 0:
            fpr = np.repeat(np.nan, fps.shape)
        else:
            fpr = fps / fps[-1]

        if tps[-1] <= 0:
            tpr = np.repeat(np.nan, tps.shape)
        else:
            tpr = tps / tps[-1]

        auc = np.trapz(fpr, tpr)

        return auc, fpr, tpr, thresholds
    
    
    def save_model(self, filepath):
        '''
        Function: 
            Saves the model as a .h5 file that stores each layers neurons, weights, bias, loss fn and
            activation function.

            ** Note: Do not add .h5 to the end of your filepath. This will be added automatically.
        '''
        filename = str(uuid.uuid4().hex) + ".h5"
        final_path = filepath + "/" + filename

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
            "Softmax": Softmax,
            "tanh": Tanh,
            "LeakyReLU": LeakyReLU, 
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



   