import torch
import torch.nn.functional as F

# Define the RNN parameters from the given image
b = torch.tensor([[-1.0], [1.0]])
c = torch.tensor([[0.5], [-0.5]])
W = torch.tensor([[1.0, -1.0], [0.0, 2.0]])
U = torch.tensor([[-1.0, 0.0], [1.0, -2.0]])
V = torch.tensor([[-2.0, 1.0], [-1.0, 0.0]])

# Define the input sequence as given in the image
x = {1: torch.tensor([[1.0], [0.0]]), 
     2: torch.tensor([[0.50], [0.25]]), 
     3: torch.tensor([[0.0], [1.0]])}

# Initial hidden state h(0) is assumed to be zero
h = {0: torch.zeros(2, 1)}

# Function to compute the RNN step
def rnn_step(x_t, h_tm1, U, W, b):
    return torch.tanh(U @ x_t + W @ h_tm1 + b)

# Function to compute the output of the network
def output_step(h_t, V, c):
    return V @ h_t + c

# Placeholder for the outputs
y_hat = {}

# Compute the RNN outputs for each time step and the custom loss
loss = 0
for t in range(1, 4):
    h[t] = rnn_step(x[t], h[t-1], U, W, b)
    y_hat[t] = output_step(h[t], V, c)
    # Custom loss for the given time step
    loss_t = (y_hat[t][1] - 0.25)**2 - torch.log(y_hat[t][0])
    loss += loss_t

# Outputs and loss for each time step
for t in range(1, 4):
    print(f"y_hat({t}): {y_hat[t].T}")

print("Custom loss:", loss.item())


# Define the RNN parameters again 
b = torch.tensor([[-1.0], [1.0]], requires_grad=False)
c = torch.tensor([[0.5], [-0.5]])
W = torch.tensor([[1.0, -1.0], [0.0, 2.0]])
U = torch.tensor([[-1.0, 0.0], [1.0, -2.0]])
V = torch.tensor([[-2.0, 1.0], [-1.0, 0.0]])

# Define the input sequence
x = [torch.tensor([[1.0], [0.0]]), 
     torch.tensor([[0.50], [0.25]]), 
     torch.tensor([[0.0], [1.0]])]

epsilon = 0.000025

# Function to calculate the loss
def calculate_loss(b, x):
    h = torch.zeros(2, 1)
    for x_t in x:
        h = torch.tanh(W @ h + U @ x_t + b)
    y = V @ h + c
    loss = (y[1] - 0.25)**2 - torch.log(y[0])
    return loss

# Calculate gradient using central difference
grad_b = torch.zeros_like(b)
for i in range(len(b)):
    b_plus = b.clone()
    b_plus[i] += epsilon
    loss_plus = calculate_loss(b_plus, x)

    b_minus = b.clone()
    b_minus[i] -= epsilon
    loss_minus = calculate_loss(b_minus, x)

    grad_b[i] = (loss_plus - loss_minus) / (2 * epsilon)

# Output the estimated gradients
print("fist grad",grad_b)

b = torch.tensor([[-1.0], [1.0]], requires_grad=True)


h = torch.zeros(2, 1, requires_grad=True)  # Initial hidden state
loss = 0
for t in range(3):
    h = torch.tanh(W @ h + U @ x[t] + b)
    y_hat = V @ h + c
    if t == 2:  # Compute loss at the last time step
        loss = (y_hat[1] - 0.25)**2 - torch.log(y_hat[0])

# Perform backpropagation to compute gradients with respect to b
loss.backward()


# Print out the gradients and loss
print("last grad",b.grad)
print(loss.item())

learning_rate = 0.005

with torch.no_grad():  # Temporarily set all the requires_grad flag to false
    b -= learning_rate * b.grad

# Reset gradients for the next optimization step
b.grad.zero_()

# Print the updated values of b
print("updated values", b)

c = torch.tensor([[0.5], [-0.5]])
W = torch.tensor([[1.0, -1.0], [0.0, 2.0]])
U = torch.tensor([[-1.0, 0.0], [1.0, -2.0]])
V = torch.tensor([[-2.0, 1.0], [-1.0, 0.0]])

x = [torch.tensor([[1.0], [0.0]]), 
     torch.tensor([[0.50], [0.25]]), 
     torch.tensor([[0.0], [1.0]])]

# Re-initialize hidden state
h = torch.zeros(2, 1)

# Compute the RNN outputs for each time step and the custom loss with the updated 'b'
new_loss = 0
for t in range(3):
    h = torch.tanh(W @ h + U @ x[t] + b)
    y_hat = V @ h + c
    if t == 2:  # Compute loss at the last time step
        new_loss = (y_hat[1] - 0.25)**2 - torch.log(y_hat[0])

# Print the new loss
print("New loss with updated b:", new_loss.item())
