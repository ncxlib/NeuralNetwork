from neuralnetwork.layers.fullyconnectedayer import FullyConnectedLayer
import numpy as np
from neuralnetwork.activations.activations import relu

def relu(x: np.ndarray) -> np.ndarray:
    one = np.array(1.0, dtype=x.dtype)
    return one / (one + np.exp(-x))


def main():
    inputs = np.array([0.5, -0.2, 0.3])
    fcl = FullyConnectedLayer(n_inputs=3, n_neurons=5, activation_fn=relu)

    output = fcl.forward_propagation(inputs)
    
    print("Layer 1", output)

if __name__ == "__main__":
    main()