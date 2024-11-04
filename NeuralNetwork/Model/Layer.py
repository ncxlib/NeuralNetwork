class Layer:
    def __init__(self, activation_fn):
        self.activation_fn = activation_fn
        self.layer = []
