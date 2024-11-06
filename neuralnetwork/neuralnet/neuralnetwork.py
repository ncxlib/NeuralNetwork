from neuralnetwork.layers.fullyconnectedayer import FullyConnectedLayer

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_propagation(inputs)
        return inputs

    def backward_propagate(self, y_orig, y_pred):
        for layer in reversed(self.layers):
            layer.back_propagation(y_orig, y_pred)
    

    def train(self, inputs, y_orig, epochs):
        for epoch in range(epochs):
            y_pred = self.forward_propagation(inputs)

            loss = self.layers[-1].calculate_loss(y_pred, y_orig)
            print(f"Epoch {epoch + 1}, Loss: {loss}")

            self.backward_propagate(y_orig, y_pred)

    def predict(self, inputs):
        return self.forward_propagation(inputs)

    # TODO: Build out the neural net && all methods..

    # compile(optimizer, type_of_loss=MSE)
    # fit(features (x), labels(y), epochs, validation_set(x (validation set), y (validation set)))
    # back propagation happens here
    # evaluate(test_x, test_y) --> outputs loss && accuracy

    # neural net class (model) takes in array of layers
    # each base layer takes in (n_neurons, activation_fn)
