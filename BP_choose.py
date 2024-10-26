import numpy as np


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the sigmoid function (for backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)


# Define the neural network for the second structure
class SecondStructureNeuralNetwork:
    def __init__(self):
        # Initialize weights randomly
        # 2 inputs (x1, x2), 1 hidden neuron (h2), 1 output neuron (y)
        self.weights_input_hidden = np.random.rand(2, 1)  # 2 inputs -> 1 hidden neuron (h2)
        self.weights_hidden_output = np.random.rand(1, 1)  # 1 hidden neuron (h2) -> 1 output neuron (y)

        # Initialize biases randomly
        self.bias_hidden = np.random.rand(1, 1)  # Bias for the hidden neuron
        self.bias_output = np.random.rand(1, 1)  # Bias for the output neuron

        # Bias weights connecting bias node to hidden and output layers
        self.weight_bias_hidden = np.random.rand(1, 1)  # Bias to hidden neuron
        self.weight_bias_output = np.random.rand(1, 1)  # Bias to output neuron

    # Forward pass (input to hidden to output)
    def forward(self, X):
        # Bias node output is always 1
        bias = np.ones((X.shape[0], 1))

        # Calculate the hidden layer activation
        self.hidden_input = np.dot(X, self.weights_input_hidden) + bias * self.weight_bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # Calculate the final output
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + bias * self.weight_bias_output
        self.final_output = sigmoid(self.output_input)
        return self.final_output

    # Backpropagation to update weights and biases
    def backward(self, X, y, learning_rate):
        # Calculate the error at the output layer
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)

        # Calculate the error for the hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Bias node output is always 1
        bias = np.ones((X.shape[0], 1))

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.weight_bias_output += np.sum(output_delta * bias, axis=0) * learning_rate

        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.weight_bias_hidden += np.sum(hidden_delta * bias, axis=0) * learning_rate

    # Train the neural network
    def train(self, X, y, iterations, learning_rate):
        for _ in range(iterations):
            self.forward(X)  # Forward pass
            self.backward(X, y, learning_rate)  # Backward pass (update weights)

    # Predict the output for new input
    def predict(self, X):
        return self.forward(X)


# Define the XOR problem inputs and outputs
X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])  # XOR inputs
y = np.array([[-1], [1], [1], [-1]])  # XOR expected outputs

# Create the neural network and train it
nn = SecondStructureNeuralNetwork()
nn.train(X, y, iterations=10000, learning_rate=0.1)

# Test the neural network with the XOR inputs
for i in range(len(X)):
    print(f"Input: {X[i]} - Predicted Output: {nn.predict(X[i])}")
