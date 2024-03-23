import numpy as np
import re

# Replace 'path/to/your/file.m' with the actual path to your MATLAB file
file_path = './hw2_softmax_weights.m'
# Initialize variables
softmax_weight = None
softmax_bias = None


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)


# Read the MATLAB file
with open(file_path, 'r') as file:
    file_content = file.read()

    # Regular expression to find the assignment of arrays
    weight_match = re.search(
        r'softmax_weight\s*=\s*\[(.*?)\];', file_content, re.DOTALL)
    bias_match = re.search(
        r'softmax_bias\s*=\s*\[(.*?)\];', file_content, re.DOTALL)

    if weight_match:
        # Extract and convert the string of numbers to a numpy array
        weight_str = weight_match.group(1).replace('\n', ' ')
        softmax_weight = np.fromstring(weight_str, sep=' ')

    if bias_match:
        # Extract and convert the string of numbers to a numpy array
        bias_str = bias_match.group(1).replace('\n', ' ')
        softmax_bias = np.fromstring(bias_str, sep=' ')

# Check the results
if softmax_weight is not None:
    print("Softmax Weights Extracted: ", softmax_weight)
if softmax_bias is not None:
    print("Softmax Bias Extracted: ", softmax_bias)


print("Number of weights", len(softmax_weight))
print("Number of biases", len(softmax_bias))

# Calculate the number of inputs
if softmax_weight is not None:
    num_inputs = len(softmax_weight) // len(softmax_bias)
    print("Number of inputs:", num_inputs)


asample = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.606391429901123, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9543248414993286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1392189860343933, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.836493968963623, 0.0, 0.12610933184623718,
           0.0, 0.0, 0.0, 0.0843304991722107, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4557386338710785, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3026450276374817, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6092420816421509, 0.23424609005451202, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# Compute the logits
logits = np.dot(asample, softmax_weight.reshape(
    (num_inputs, -1))) + softmax_bias

# Compute the probabilities using softmax
probabilities = softmax(logits)
predicted_class = np.argmax(probabilities)

print("Probabilities 1:", probabilities)
print("Predicted Class 1:", predicted_class)

correct_label = 5
y_true = np.zeros(20)  # Assuming 20 classes
y_true[correct_label] = 1


gradient_logits = probabilities - y_true

# Assuming a learning rate of 0.1
learning_rate = 0.1

# Compute gradients for weights and biases
# Gradient for weights is the outer product of the gradient with respect to logits and the input sample
gradient_weights = np.outer(asample, gradient_logits)

# Gradient for biases is directly the gradient of the logits (since bias contributes directly to each logit)
gradient_biases = gradient_logits

# Update weights and biases
softmax_weight = softmax_weight.reshape(
    (num_inputs, -1)) - learning_rate * gradient_weights
softmax_bias = softmax_bias - learning_rate * gradient_biases

logits = np.dot(asample, softmax_weight.reshape(
    (num_inputs, -1))) + softmax_bias

# Compute the probabilities using softmax
probabilities = softmax(logits)
# Classify the example
predicted_class = np.argmax(probabilities)

print("Probabilities 2:", probabilities)
print("Predicted Class 2:", predicted_class)

softmax_weight_reshaped = softmax_weight.reshape((num_inputs, -1))

# Compute the update values for weights and biases
weight_updates = learning_rate * np.outer(asample, gradient_logits)
bias_updates = learning_rate * gradient_logits

# Apply updates to weights and biases
updated_weights = softmax_weight_reshaped - weight_updates
updated_biases = softmax_bias - bias_updates

# Calculate changes
weight_changes = updated_weights - softmax_weight_reshaped
bias_changes = updated_biases - softmax_bias

# Determine the counts
weights_increased = np.sum(weight_changes > 0.00001)
weights_decreased = np.sum(weight_changes < -0.00001)
weights_unchanged = np.sum(np.abs(weight_changes) <= 0.00001)

biases_increased = np.sum(bias_changes > 0.00001)
biases_decreased = np.sum(bias_changes < -0.00001)
biases_unchanged = np.sum(np.abs(bias_changes) <= 0.00001)

print(
    f"Weights Increased: {weights_increased}, Decreased: {weights_decreased}, Unchanged: {weights_unchanged}")
print(
    f"Biases Increased: {biases_increased}, Decreased: {biases_decreased}, Unchanged: {biases_unchanged}")


def compute_input_gradient(asample, probabilities, y_true, softmax_weight):

    prob_diff = probabilities - y_true

    print("Shape of softmax_weight.T:", softmax_weight.shape)

    print("Shape of prob_diff:", prob_diff.shape)  # Should be (20,)

    gradient_wrt_input = np.dot(softmax_weight, prob_diff)

    return gradient_wrt_input


# Compute the gradient of the loss with respect to the input sample.
epsilon = 0.01  # Starting small
max_iterations = 100
tolerance = 1e-6  # Tolerance to avoid too small updates

for i in range(max_iterations):
    # Recalculate everything based on current adversarial_sample
    logits = np.dot(asample, softmax_weight.reshape((-1, 20))) + softmax_bias
    probabilities = softmax(logits)
    predicted_class = np.argmax(probabilities)

    # Check if the desired class is achieved
    if predicted_class == correct_label:
        print("Achieved target class with epsilon:", epsilon)
        break

    print("Shape of softmax_weight before transpose:", softmax_weight.shape)
    softmax_weight_T = softmax_weight.T
  # If not, compute the gradient and update the sample
    gradient_wrt_input = compute_input_gradient(
        asample, probabilities, y_true, softmax_weight)

   # Ensure the gradient step is significant enough
    if np.linalg.norm(gradient_wrt_input) < tolerance:
        print("Gradient too small, increasing epsilon")
        epsilon *= 10  # Increase epsilon to enforce a more substantial update
        continue

    # Update asample in the direction that should increase the probability of the correct class
    asample = asample - epsilon * gradient_wrt_input / \
        np.linalg.norm(gradient_wrt_input)

    # Optionally, check the size of the perturbation and stop if it exceeds a predefined threshold

# Recompute final logits and probabilities for the updated adversarial sample
    logits_adversarial = np.dot(
        asample, softmax_weight.reshape((-1, 20))) + softmax_bias
    probabilities_adversarial = softmax(logits_adversarial)
    predicted_class_adversarial = np.argmax(probabilities_adversarial)

    print("Final Adversarial Probabilities:", probabilities_adversarial)
    print("Final Adversarial Predicted Class:", predicted_class_adversarial)
