import numpy as np

from models.dl.activations import activation_functions

# Description: Multilayer Perceptron (MLP) implementation using NumPy.


class MLP:
    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(layers) - 1):
            weight = np.random.randn(layers[i], layers[i + 1]) * 0.1
            bias = np.zeros((1, layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        self.a = [X]
        self.z = []

        for i in range(len(self.weights)):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)
            a = activation_functions[self.activations[i]][0](z)
            self.a.append(a)

        return self.a[-1]

    def backward(self, X, y, learning_rate):
        m = y.shape[0]
        deltas = [self.a[-1] - y]

        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(deltas[-1], self.weights[i + 1].T) * \
                activation_functions[self.activations[i]][1](self.a[i + 1])
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * \
                np.dot(self.a[i].T, deltas[i]) / m
            self.biases[i] -= learning_rate * \
                np.sum(deltas[i], axis=0, keepdims=True) / m

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            if (epoch + 1) % 100 == 0:
                loss = self.calculate_loss(y, output)
                print(f'Epoch {epoch + 1}, Loss: {loss}')

    def calculate_loss(self, y, output):
        return -np.mean(y * np.log(output + 1e-9))

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Example usage:
# layers = [4, 5, 3]  # 4 input neurons, 5 neurons in the hidden layer, 3 output neurons
# activations = ['relu', 'softmax']
# mlp = MLP(layers, activations)
# mlp.train(X_train, y_train, epochs=1000, learning_rate=0.01)
# predictions = mlp.predict(X_test)
