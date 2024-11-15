from ncxlib.neuralnetwork.layers.layer import Layer
import numpy as np
from typing import Optional
from ncxlib.util import log
from ncxlib.neuralnetwork.optimizers.optimizer import Optimizer
from ncxlib.neuralnetwork.optimizers.sgd import SGD
from ncxlib.neuralnetwork.activations.activation import Activation
from ncxlib.neuralnetwork.activations.relu import ReLU
from ncxlib.neuralnetwork.losses import LossFunction, MeanSquaredError
from ncxlib.neuralnetwork.initializers import Initializer, HeNormal, Zero

class FullyConnectedLayer(Layer):
    def __init__(
        self,
        n_neurons: Optional[int] = None,
        n_inputs: Optional[int] = None,
        activation: Optional[Activation] = ReLU,
        optimizer: Optional[Optimizer] = SGD(),
        loss_fn: Optional[LossFunction] = MeanSquaredError,
        initializer: Optional[Initializer] = HeNormal(),
        weights_initializer: Optional[Initializer] = HeNormal(),
        bias_initializer: Optional[Initializer] = Zero(),
        name: Optional[str] = ""
    ):
        super().__init__(n_neurons, n_inputs, activation, optimizer, loss_fn, name=name)

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

        # calculate weighted sum: X W' + b (broadcast b automatic with numpy)
        weighted_sum = np.dot(self.inputs, self.W.T) + self.b.T

        # activate each neuron with self.activation function
        activated =  self.activation.apply(weighted_sum)

        # if saving: (bad var name i guess)
        if not no_save:
            self.z = weighted_sum
            self.activated = activated

        return activated
    
    def back_propagation(self, next_layer: Layer, learning_rate: float) -> np.ndarray:

        da_dz = self.activation.derivative(self.z) 

        dl_da = next_layer.gradients @ next_layer.old_W

        # size: batch_size x n_neurons
        dl_dz = dl_da * da_dz

        self.old_W = self.W.copy()

        # size batch_size x n_inputs
        dz_dw = self.inputs

        # size multiply  n_neurons x batch_size times batch_size x n_inputs = n_neurons x n_inputs
        dl_dw = dl_dz.T @ dz_dw

        dl_db = np.sum(dl_dz, axis=0, keepdims=True)

        # size: n_neurons x 1
        dl_db = dl_db.T

        self.gradients = dl_dz

        self.W, self.b = self.optimizer.apply(self.W, dl_dw, self.b, dl_db)