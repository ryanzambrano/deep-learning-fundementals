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
g_x = torch.sin(A / (x**B + 1e-6))

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
        # anotha one
        self.fc3 = nn.Linear(hidden_layer, hidden_layer)

        self.fc4 = nn.Linear(hidden_layer, hidden_layer)
        self.fc5 = nn.Linear(hidden_layer, hidden_layer)
        self.fc6 = nn.Linear(hidden_layer, hidden_layer)
        self.fc7 = nn.Linear(hidden_layer, hidden_layer)
        # last hidden layer to output layer
        self.fc8 = nn.Linear(hidden_layer, output_layer)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function after first layer
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        # Activation function after second layer
        # No activation after last layer (depends on the problem)
        x = self.fc8(x)
        return x


net = Net(input_layer=1, hidden_layer=400, output_layer=1)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

epochs = 1100
for epoch in range(epochs):
    optimizer.zero_grad()   # Zero the gradient buffers
    output = net(x)         # Pass the input through the network
    loss = criterion(output, g_x)  # Compute the loss

    errors = torch.abs(output - g_x)
    max_errors.append(torch.max(errors).item())
    avg_errors.append(torch.mean(errors).item())

    loss.backward()         # Backpropagate the loss
    optimizer.step()        # Update the weights

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

with torch.no_grad():  # We don't need gradients for evaluation
    predicted = net(x_normalized)
fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns

# First subplot: True function vs. NN approximation
axs[0].plot(x.numpy(), g_x.numpy(), label='True Function')
axs[0].plot(x.numpy(), predicted.numpy(),
            label='NN Approximation', linestyle='--')
axs[0].legend()
axs[0].set_xlabel('x')
axs[0].set_ylabel('g(x)')
axs[0].set_title('Function Approximation')

# Second subplot: Training loss, Max error, and Average error over epochs
axs[1].plot(losses, label='Training Loss')
axs[1].plot(max_errors, label='Max Error')
axs[1].plot(avg_errors, label='Average Error')
axs[1].legend()
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Error')
axs[1].set_title('Training Dynamics')

# Display the figure with both subplots
plt.show()
