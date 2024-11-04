from Layers.FullyConnectedLayer import FullyConnectedLayer
import numpy as np

def main():
    inputs = np.array([0.5, -0.2, 0.3])
    fcl = FullyConnectedLayer(n_inputs=3, n_neurons=5, activation_fn="relu")

    output = fcl.forward_propagation(inputs)
    
    print("Layer 1", output)

if __name__ == "__main__":
    main()