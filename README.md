# NCxLib: A Lightweight Neural Network Library in Python

ncxlib is a lightweight and easy-to-use neural network library built in Python. It provides a simple API for constructing and training neural networks, along with tools for data preprocessing and generation.

## Features

* **Modular Design:** Easily build custom neural networks by combining different layers, activation functions, and loss functions.
* **Data Handling:** Includes data loaders for CSV and image data, with preprocessing capabilities like scaling and grayscaling.
* **Training and Evaluation:** Train your networks with various optimization algorithms and evaluate their performance.
* **Extensible:**  Add your own custom layers, activations, and loss functions to expand the library's functionality.

## Installation

```bash
pip install ncxlib
```

## Getting Started
Here's a quick example of how to use ncxlib to create and train a simple neural network:

```python
import numpy as np
from ncxlib.dataloaders import CSVDataLoader
from ncxlib.preprocessing import MinMaxScaler
from ncxlib.neuralnetwork import NeuralNetwork
from ncxlib.neuralnetwork.layers import FullyConnectedLayer
from ncxlib.neuralnetwork.activations import Sigmoid, ReLU
from ncxlib.neuralnetwork.losses import BinaryCrossEntropy
from ncxlib.generators import generate_training_data, train_test_split

# Generate synthetic data
generate_training_data(
    num_samples=100,
    num_features=3,
    to_csv=True,
)

# Load data with preprocessing
loader = CSVDataLoader("training_data.csv", preprocessors=[MinMaxScaler()])
X, y = loader.get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the neural network architecture
network = NeuralNetwork(
    [
        FullyConnectedLayer(n_neurons=4, activation=ReLU(), name="first_hidden"),
        FullyConnectedLayer(n_neurons=6, activation=ReLU(), name="second_hidden"),
        FullyConnectedLayer(n_neurons=2 , activation=Sigmoid(), name="output"),
    ],
    loss_fn=BinaryCrossEntropy() 
)

# Train the network
network.train(X_train, y_train, epochs=100, learning_rate=0.01)

# Evaluate the network
network.evaluate(X_test, y_test)
```


## License
This project is licensed under the [MIT License](LICENSE)