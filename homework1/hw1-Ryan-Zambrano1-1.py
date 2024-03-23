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
plt.plot(input_range, gradient, label='Gradient of ReLU', color='blue')
plt.plot(input_range, actual_relu, label='ReLU Function', color='green')

# Fast Learning Region - for positive inputs where ReLU is active
plt.fill_between(input_range, 0, gradient, where=(input_range > 0),
                 color='red', alpha=0.3, label='Fast Learning Region (|gradient| = 1)')

# Hypothetical Slow Learning Region - Illustrated across the plot for demonstration
# Note: This is purely illustrative and does not represent a real characteristic of ReLU.
plt.fill_between(input_range, 0, 0.01, color='purple', alpha=0.3,
                 label='Hypothetical Slow Learning Region (0 < |gradient| < 0.01)')

# Inactive Learning Region - for negative inputs where ReLU's gradient is 0
plt.fill_between(input_range, 0, where=(input_range <= 0), color='orange',
                 alpha=0.3, label='Inactive Learning Region (|gradient| = 0)')

# Active Learning Region - Not directly applicable to ReLU, but including for completeness
plt.axhline(y=0.01, color='green', linestyle='--',
            label='Active Learning Region (0.01 <= |gradient| <= 0.99)')

plt.xlabel('Input')
plt.ylabel('Value / Gradient')
plt.title('ReLU Activation Function and Learning Regions')
plt.legend()
plt.grid(True)
plt.show()
