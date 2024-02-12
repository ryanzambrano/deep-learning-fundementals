import numpy as np
import matplotlib.pyplot as plt


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return np.where(z >= 0, 1, 0)


# Define the range of input values
input_range = np.linspace(-5, 5, 100)

# Calculate gradients
gradient = relu_derivative(input_range)

actual_relu = relu(input_range)

# Plot the gradient
plt.figure(figsize=(10, 6))
plt.plot(input_range, gradient, label='Gradient of ReLUU', color='blue')
plt.plot(input_range, actual_relu,
         label='Linear function of ReLU', color='green')
plt.axhline(y=0.99, color='red', linestyle='--',
            label='Fast Learning Region (|gradient| > 0.99)')
plt.axhline(y=0.01, color='green', linestyle='--',
            label='Active Learning Region (0.01 <= |gradient| <= 0.99)')
plt.axhline(y=0, color='orange', linestyle='--',
            label='Inactive Learning Region (|gradient| = 0)')
plt.xlabel('Input')
plt.ylabel('Gradient')
plt.title('Gradient of ReLU Activation Function')
plt.legend()
plt.grid(True)
plt.show()
