import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


A, B = 1, 2
# Reshape and ensure it's a float
x = torch.linspace(-2, 2, 200).view(-1, 1).float()
x_mean = x.mean()
x_std = x.std()
x_normalized = (x - x_mean) / x_std

# Avoid division by zero by adding a small value to x in the denominator
g_x = torch.sin(np.pi/4*x)

losses = []
max_errors = []
avg_errors = []


class Net(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer):
        super(Net, self).__init__()

        # Input layer to first hidden layer
        self.fc1 = nn.Linear(input_layer, hidden_layer)
        # First hidden layer to second hidden layer
        # Input layer to second hidden layerlayer) # Input layer to second hidden layerlayer)

        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, hidden_layer)
        # last hidden layer to output layer
        self.fc4 = nn.Linear(hidden_layer, output_layer)

    def forward(self, x, return_patterns=False):

        patterns = []

        x1 = self.fc1(x)
        p1 = (x1 > 0).int()
        x1 = torch.relu(x1)
        patterns.append(p1)

        x2 = self.fc2(x1)
        p2 = (x2 > 0).int()
        x2 = torch.relu(x2)
        patterns.append(p2)

        x3 = self.fc3(x2)
        p3 = (x3 > 0).int()
        x3 = torch.relu(x3)
        patterns.append(p3)

        # Activation function after second layer
        # No activation after last layer (depends on the problem)
        x4 = self.fc4(x3)
        if return_patterns:
            return x4, patterns
        else:
            return x4


net = Net(input_layer=1, hidden_layer=64, output_layer=1)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

epochs = 1100
for epoch in range(epochs):
    optimizer.zero_grad()   # Zero the gradient buffers
    output = net(x)         # Pass the input through the network
    loss = criterion(output, g_x)  # Compute the loss
    losses.append(loss.item())

    errors = torch.abs(output - g_x)
    max_errors.append(torch.max(errors).item())
    avg_errors.append(torch.mean(errors).item())

    loss.backward()         # Backpropagate the loss
    optimizer.step()        # Update the weights

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

with torch.no_grad():
    predicted, patterns = net(x, return_patterns=True)

    # Convert to numpy for easier handling with matplotlib
    first_layer_patterns = patterns[0].numpy()
    pattern_integers = first_layer_patterns.dot(
        2**torch.arange(first_layer_patterns.shape[1] - 1, -1, -1))


# `x` is your input range, and `first_layer_patterns` now contains a binary matrix of activations for the first layer
num_neurons = first_layer_patterns.shape[1]
# fig, ax = plt.subplots(figsize=(10, 6))
fig, axs = plt.subplots(1, 3, figsize=(30, 6))

# Scatter plot for activation regions
scatter = axs[2].scatter(x.numpy().flatten(), np.zeros_like(x.numpy()).flatten(),
                         c=pattern_integers, cmap='viridis', s=2, alpha=0.5)  # Added alpha for better visualization

# Create a colorbar to interpret the colors
cbar = plt.colorbar(scatter, ax=axs[2])
cbar.set_label('Activation Pattern Code')

# Set the labels and title
axs[2].set_xlabel('Input x')
# Hides y-axis ticks as they're not meaningful in this context
axs[2].set_yticks([])
axs[2].set_title('Activation Regions for First Layer Neurons')

# Annotating activation regions with binary patterns# Assuming you've already plotted the scatter plot as before
unique_patterns, indices = np.unique(pattern_integers, return_index=True)


# First subplot: True function vs. NN approximation
axs[0].plot(x.numpy(), g_x.numpy(), label='True Function')
axs[0].plot(x.numpy(), predicted.numpy(),
            label='NN Approximation', linestyle='--')
axs[0].legend()
axs[0].set_xlabel('x')
axs[0].set_ylabel('g(x)')
axs[0].set_title('Function Approximation')

# Second subplot: Training loss, Max error, and Average error over epochs
axs[1].plot(losses, label='Training error', linestyle='--')
axs[1].plot(max_errors, label='Max Error')
axs[1].plot(avg_errors, label='Average Error')
axs[1].legend()
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Error')
axs[1].set_title('Training Dynamics')

plt.show()
