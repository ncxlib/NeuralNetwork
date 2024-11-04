from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    def __init__(self, activation_fn: str):
        self.activation_fn = placeholder.get(activation_fn)
        if self.activation_fn is None:
            raise ValueError(
                "Missing activation function. Cannot be empty. Example: activation_fn=Relu"
            )

    @abstractmethod
    def initialize_params(self):
        pass

    @abstractmethod
    def forward_propogation(self):
        pass
