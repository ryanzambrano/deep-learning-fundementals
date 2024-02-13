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

# Classify the example
predicted_class = np.argmax(probabilities)

print("Probabilities:", probabilities)
print("Predicted Class:", predicted_class)
