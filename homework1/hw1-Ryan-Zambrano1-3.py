import numpy as np
import matplotlib.pyplot as plt


def piece_wise(z):
    # Vectorized condition
    return np.where(z > 1, 0.1 * z + 0.9, np.where(z >= -1, z, 0.1 * z - 0.9))


def piece_wise_derivative(z):
    # Vectorized condition for the derivative
    return np.where(z > 1, 0.1, np.where(z >= -1, 1, 0.1))
    # return sigmoid(z) * (1 - sigmoid(z))


# Define the range of input values
input_range = np.linspace(-5, 5, 100)

# Calculate gradients
gradient = piece_wise_derivative(input_range)

actual_relu = piece_wise(input_range)

# Plot the gradient
plt.figure(figsize=(10, 6))
plt.plot(input_range, gradient,
         label='Gradient of Piece Wise Function', color='blue')
plt.plot(input_range, actual_relu,
         label='Piece Wise Activation Function', color='green')
plt.axhline(y=0.99, color='red', linestyle='--',
            label='Fast Learning Region (|gradient| > 0.99)')
plt.axhline(y=0.01, color='green', linestyle='--',
            label='Active Learning Region (0.01 <= |gradient| <= 0.99)')
plt.axhline(y=0, color='orange', linestyle='--',
            label='Inactive Learning Region (|gradient| = 0)')
plt.xlabel('Input')
plt.ylabel('Gradient')
plt.title('Gradient of Piece Wise Activation Function')
plt.legend()
plt.grid(True)
plt.show()
