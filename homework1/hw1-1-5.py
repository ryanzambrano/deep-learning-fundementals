import numpy as np
import matplotlib.pyplot as plt


def elu(z, alpha=0.05):
    return np.where(z >= 0, z, alpha * (np.exp(z) - 1))


def elu_derivative(z, alpha=0.05):
    return np.where(z >= 0, 1, alpha * np.exp(z))


# Define the range of input values
input_range = np.linspace(-5, 5, 100)

# Calculate gradients
gradient = elu_derivative(input_range)

elu_func = elu(input_range)

# Plot the gradient
plt.figure(figsize=(10, 6))
plt.plot(input_range, gradient,
         label='Gradient of ELU Function', color='blue')
plt.plot(input_range, elu_func,
         label='ELU Activation Function', color='green')
plt.axhline(y=0.99, color='red', linestyle='--',
            label='Fast Learning Region (|gradient| > 0.99)')
plt.axhline(y=0.01, color='green', linestyle='--',
            label='Active Learning Region (0.01 <= |gradient| <= 0.99)')
plt.axhline(y=0, color='orange', linestyle='--',
            label='Inactive Learning Region (|gradient| = 0)')
plt.xlabel('Input')
plt.ylabel('Gradient')
plt.title('Gradient of ELU Activation Function')
plt.legend()
plt.grid(True)
plt.show()
