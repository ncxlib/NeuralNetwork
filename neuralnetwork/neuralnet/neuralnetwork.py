from neuralnetwork.layers.fullyconnectedayer import FullyConnectedLayer
from neuralnetwork.losses.losses import MSE

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def build_layer(self,n_inputs, n_neurons, activation, optimizer, layer_type : FullyConnectedLayer):
        layer = FullyConnectedLayer(n_inputs, n_neurons, activation, optimizer, layer_type)
        self.add_layer(layer)

    def forward_propagation(self, input_vector):
        for layer in self.layers:
            input_vector = layer.forward_propagation(input_vector)
        return input_vector

    def backward_propagate(self, y_orig, y_pred):
        for layer in reversed(self.layers):
            layer.back_propagation(y_orig, y_pred)
    

    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(inputs)):

                input_vector = inputs[i]
                y_true = targets[i]

                y_pred = self.forward_propagation(input_vector)

                loss = MSE(y_pred, y_true)
                total_loss += loss

                self.back_propagation(y_true, y_pred)

            average_loss = total_loss / len(inputs)
            print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

    def predict(self, inputs):
        return self.forward_propagation(inputs)

    # TODO: Build out the neural net && all methods..

    # compile(optimizer, type_of_loss=MSE)
    # fit(features (x), labels(y), epochs, validation_set(x (validation set), y (validation set)))
    # back propagation happens here
    # evaluate(test_x, test_y) --> outputs loss && accuracy

    # neural net class (model) takes in array of layers
    # each base layer takes in (n_neurons, activation_fn)
