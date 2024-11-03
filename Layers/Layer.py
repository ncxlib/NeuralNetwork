from ABC import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, activation_fn=None):
        self.activation_fn = activation_fn

    @abstractmethod
    def initialize_params(self):
        pass

    @abstractmethod
    def forward_propogation(self):
        pass