import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


# Define the range of input values
input_range = np.linspace(-5, 5, 100)

# Calculate gradients
gradient = sigmoid_derivative(input_range)

actual_relu = sigmoid(input_range)

# Plot the gradient
plt.figure(figsize=(10, 6))
plt.plot(input_range, gradient,
         label='Gradient of Sigmoid Function', color='blue')
plt.plot(input_range, actual_relu,
         label='Sigmoid Function', color='green')
plt.axhline(y=0.99, color='red', linestyle='--',
            label='Fast Learning Region (|gradient| > 0.99)')
plt.axhline(y=0.01, color='green', linestyle='--',
            label='Active Learning Region (0.01 <= |gradient| <= 0.99)')
plt.axhline(y=0.01, color='purple', linestyle='--',
            label='Slow Learning Region (0 < |gradient| < 0.01)', zorder=5)
plt.axhline(y=0, color='orange', linestyle='--',
            label='Inactive Learning Region (|gradient| = 0)')
plt.xlabel('Input')
plt.ylabel('Gradient')
plt.title('Gradient of Sigmoid Activation Function')
plt.legend()
plt.grid(True)
plt.show()
