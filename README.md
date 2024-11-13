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

# External imporst
import numpy as np

# Util imports
from ncxlib import generators, dataloaders
from ncxlib.util import train_test_split

# Neural network imports
from ncxlib.neuralnetwork import optimizers, losses
from ncxlib.preprocessing import MinMaxScaler
from ncxlib.neuralnetwork import NeuralNetwork, FullyConnectedLayer
from ncxlib.neuralnetwork import activations
from ncxlib.neuralnetwork.initializers import HeNormal, Zero


# ------- Generate some data using generators -------
gemerators.generate_training_data(to_csv=True)

# ------- Load data from generated csv and split it into train and test -------
loader = dataloaders.CSVDataLoader("training_data.csv")
X, y = loader.get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y)


# ------- Configure model layers -------
model = NeuralNetwork([
    FullyConnectedLayer(
        n_neurons=3, 
        activation=activations.ReLU, 
        optimizer=optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        name="first_hidden",
        weights_initializer=HeNormal(), 
        bias_initializer=Zero()
        ),

    FullyConnectedLayer(
        n_neurons=5, 
        activation=ReLU, 
        optimizer=optimizers.SGDMomentum(momentum = 0.9), 
        name="second_hidden",
        initializer=HeNormal(),
        ),

    FullyConnectedLayer(
        n_neurons=2, 
        activation=Sigmoid, 
        optimizer=optimizers.RMSProp(decay_rate = 0.8)
        )
],
    loss_fn=losses.BinaryCrossEntropy
)

# ------- Train model and evaluate accuracy -------
model.train(X_train, y_train, epochs=20, learning_rate=0.01)
model.evaluate(X_test, y_test)

```


## License
This project is licensed under the [MIT License](LICENSE)