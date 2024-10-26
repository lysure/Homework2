import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function (for backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network
class SimpleNeuralNetwork:
    def __init__(self):
        # Initialize weights randomly for 2 inputs, 3 hidden neurons, and 1 output
        self.weights_input_hidden = np.random.rand(2, 3)  # 2 inputs -> 3 hidden neurons
        self.weights_hidden_output = np.random.rand(3, 1)  # 3 hidden neurons -> 1 output

        # Initialize biases randomly
        self.bias_hidden = np.random.rand(1, 3)
        self.bias_output = np.random.rand(1, 1)

    # Forward pass (input to hidden to output)
    def forward(self, X):
        # Calculate the hidden layer activation
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # Calculate the final output
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.output_input)
        return self.final_output

    # Backpropagation to update weights and biases
    def backward(self, X, y, learning_rate):
        # Calculate the error
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)

        # Calculate the error for the hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update the weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

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
y = np.array([[0], [1], [1], [0]])  # XOR expected outputs

# Create the neural network and train it
nn = SimpleNeuralNetwork()
nn.train(X, y, iterations=10000, learning_rate=0.1)

# Test the neural network with the XOR inputs
for i in range(len(X)):
    print(f"Input: {X[i]} - Predicted Output: {nn.predict(X[i])}")
