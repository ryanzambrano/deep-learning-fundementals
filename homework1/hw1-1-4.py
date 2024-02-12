import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def swish(z):
    return z * sigmoid(2.5 * z)


def swish_derivative(z):
    sigma_z = sigmoid(2.5 * z)
    return sigma_z + 2.5 * z * sigma_z * (1 - sigma_z)


# Define the range of input values
input_range = np.linspace(-5, 5, 100)

# Calculate gradients
gradient = swish_derivative(input_range)

swish_func = swish(input_range)

# Plot the gradient
plt.figure(figsize=(10, 6))
plt.plot(input_range, gradient,
         label='Gradient of Swish Function', color='blue')
plt.plot(input_range, swish_func,
         label='Swish Activation Function', color='green')
plt.axhline(y=0.99, color='red', linestyle='--',
            label='Fast Learning Region (|gradient| > 0.99)')
plt.axhline(y=0.01, color='green', linestyle='--',
            label='Active Learning Region (0.01 <= |gradient| <= 0.99)')
plt.axhline(y=0, color='orange', linestyle='--',
            label='Inactive Learning Region (|gradient| = 0)')
plt.xlabel('Input')
plt.ylabel('Gradient')
plt.title('Gradient of Swish Activation Function')
plt.legend()
plt.grid(True)
plt.show()
