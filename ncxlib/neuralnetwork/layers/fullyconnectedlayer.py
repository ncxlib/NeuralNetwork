from ncxlib.neuralnetwork.layers.layer import Layer
import numpy as np
from typing import Optional
from ncxlib.util import log
from ncxlib.neuralnetwork.optimizers.optimizer import Optimizer
from ncxlib.neuralnetwork.optimizers.sgd import SGD
from ncxlib.neuralnetwork.activations.activation import Activation
from ncxlib.neuralnetwork.activations.relu import ReLU
from ncxlib.neuralnetwork.losses import LossFunction, MeanSquaredError

class FullyConnectedLayer(Layer):
    def __init__(
        self,
        n_inputs: Optional[int] = None,
        n_neurons: Optional[int] = None,
        activation: Optional[Activation] = ReLU,
        optimizer: Optional[Optimizer] = SGD,
        loss_fn: Optional[LossFunction] = MeanSquaredError,
        name: Optional[str] = ""
    ):
        super().__init__(n_inputs, n_neurons, activation, optimizer, loss_fn, name=name)

    def forward_propagation(self, inputs: np.ndarray, no_save: Optional[bool] = False) -> tuple[np.ndarray, int]:
        """
        inputs:
            An array of features (should be a numpy array)

        Returns:
            An array of the output values from each neuron in the layer.

        Function:
            Performs forward propagation by calculating the weighted sum for each neuron
        and applying the activation function
        """

        
       
        self.initialize_params(inputs)

        # calculate weighted sum: Wx + b
        weighted_sum = np.dot(self.W, self.inputs) + self.b

        # activate each neuron with self.activation function
        activated =  self.activation.apply(weighted_sum)

        # if saving: (bad var name i guess)
        if not no_save:
            self.z = weighted_sum
            self.activated = activated

        return activated
    
    def back_propagation(self, next_layer: Layer, learning_rate: float) -> np.ndarray:

        da_dz = self.activation.derivative(self.z) 

        dl_dz = (next_layer.old_W.T @ next_layer.gradients) * da_dz 

        self.old_W = self.W.copy()

        dz_dw = self.inputs.T  
        dl_dw = dl_dz @ dz_dw

        self.W -= learning_rate * dl_dw
        self.b -= learning_rate * dl_dz

        self.gradients = dl_dz