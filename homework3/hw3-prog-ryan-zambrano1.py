import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define the CNN architecture
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # Reduced filters
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # Reduced filters
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Reduced filters
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(32 * 3 * 3, 64),  # Simplified FC layers
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
# Set device


# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the network and optimizer
model = MNIST_CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the network
num_epochs = 5  # Adjust the number of epochs depending on the accuracy requirement
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images, labels
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, Accuracy: {100 * correct / total}%')



def visualize_filters(model):
    # Assuming model.conv_layer is a Sequential module containing your layers
    layers = [model.conv_layer[0], model.conv_layer[3], model.conv_layer[6]]
  # Adjust indices based on your model

    for layer_idx, layer in enumerate(layers):
        if hasattr(layer, 'weight'):  # Check if the layer has the 'weight' attribute
            filters = layer.weight.data.cpu().numpy()
            filter_count = filters.shape[0]

            # Set up the subplot dimensions
            n_cols = max(8, filter_count // 2)  # Ensure at least 4 cols for visibility
            n_rows = np.ceil(filter_count / n_cols)
            fig, axes = plt.subplots(nrows=int(n_rows), ncols=int(n_cols), figsize=(n_cols, n_rows))

            for i in range(filter_count):
                ax = axes.flatten()[i]
                # Assuming the filters are single-channel (grayscale)
                filter_img = filters[i, 0, :, :]
                ax.imshow(filter_img, cmap='gray')
                ax.axis('off')

            plt.suptitle(f'Filters in Layer {layer_idx + 1}')
            plt.show()
        else:
            print(f"Skipping layer {layer_idx + 1}: Not a convolutional layer.")
 



def visualize_feature_maps(model, data_loader, target_classes=[0, 8]):
    model.eval()  # Set the model to evaluation mode
    activations = []

    with torch.no_grad():
        for images, labels in data_loader:
            # Find matching examples
            for target_class in target_classes:
                match_indices = (labels == target_class).nonzero(as_tuple=False)
                if match_indices.nelement() != 0:  # Check if there's at least one match
                    index = match_indices[0, 0]  # Take the first match
                    image = images[index:index+1]  # Add batch dimension
                    image = image.to(next(model.parameters()))  # Move image to model's device
                    
                    # Adjust the following line if necessary to capture the output after the third conv layer
                    output = model.conv_layer[:6](image)  # Assuming this captures up to the third conv layer output
                    
                    activations.append((target_class, output.cpu().squeeze()))
                    if len(activations) == len(target_classes):
                        break
            if len(activations) == len(target_classes):
                break
    
    # Plotting adjustments for better readability
    for digit, activation_maps in activations:
        fig, axes = plt.subplots(1, activation_maps.size(0), figsize=(2 * activation_maps.size(0), 2))
        fig.suptitle(f'Feature Maps for Digit {digit}', fontsize=16, y=1.1)
        for i, activation_map in enumerate(activation_maps):
            if activation_maps.size(0) > 1:
                ax = axes[i]
            else:
                ax = axes
            ax.imshow(activation_map, cmap='viridis')
            ax.axis('off')
            ax.set_title(f"Map {i+1}", pad=10)  # Adjust title padding if needed

        plt.tight_layout()  # Adjust the layout to make room for the title and subplots
        plt.show()

# Assuming `model` is your trained model and `train_loader` is your data loader
visualize_feature_maps(model, train_loader)

# Ensure the model achieves at least 95% accuracy on the training set
assert 100 * correct / total >= 95, "Accuracy did not reach 95%"

visualize_filters(model)



def shift_image(image, dx, dy):
    """
    Shifts the image by dx pixels along x-axis and dy pixels along y-axis,
    filling the empty space by clamping (repeating) the border pixels.
    """
    # Create a grid to shift the image
    grid = torch.meshgrid([torch.arange(0, image.size(-2)), torch.arange(0, image.size(-1))])
    grid = torch.stack(grid, -1).float().unsqueeze(0) + torch.tensor([dy, dx]).float()
    grid = grid / torch.tensor([image.size(-2), image.size(-1)]) * 2 - 1  # Normalize to [-1, 1]

    # Apply the grid to shift the image
    shifted_image = torch.nn.functional.grid_sample(image.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return shifted_image.squeeze(0)

# Select an image of a '7' that is classified correctly
model.eval()
for images, labels in train_loader:
    images, labels = images, labels
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    match = (labels == 7) & (preds == 7)
    if match.any():
        idx = match.nonzero(as_tuple=False)[0].item()
        seven_img = images[idx]
        break

# Shift the image
shifted_left = shift_image(seven_img, -5, 0)  # Shift 5 pixels to the left
shifted_right = shift_image(seven_img, 5, 0)  # Shift 5 pixels to the right

# Predict the shifted images
with torch.no_grad():
    pred_original = model(seven_img.unsqueeze(0))
    pred_shifted_left = model(shifted_left.unsqueeze(0))
    pred_shifted_right = model(shifted_right.unsqueeze(0))

print("Original prediction:", pred_original.argmax().item())
print("Shifted left prediction:", pred_shifted_left.argmax().item())
print("Shifted right prediction:", pred_shifted_right.argmax().item())


# Define the function to move the patch and record probabilities and labels
def occlusion_test(model, image, label, occlusion_size=(8, 8), stride=1):
    C, H, W = image.size()
    # Initialize maps with zeros
    prob_map_6 = torch.zeros((H//stride, W//stride))
    max_prob_map = torch.zeros_like(prob_map_6)
    label_map = torch.zeros_like(prob_map_6, dtype=torch.long)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Loop over the image
    for y in range(0, H - occlusion_size[1] + 1, stride):
        for x in range(0, W - occlusion_size[0] + 1, stride):
            occluded_image = image.clone().detach()
            # Apply occlusion
            occluded_image[:, y:y+occlusion_size[1], x:x+occlusion_size[0]] = 0
            occluded_image = occluded_image.unsqueeze(0)
            
            # Get model output
            output = model(occluded_image)
            probs = torch.softmax(output, dim=1)
            
            # Update maps
            prob_map_6[y//stride, x//stride] = probs[0, label]
            max_prob_map[y//stride, x//stride] = torch.max(probs)
            label_map[y//stride, x//stride] = torch.argmax(probs)
    
    return prob_map_6, max_prob_map, label_map

# Function to visualize the maps
def plot_map(map_data, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(map_data, cmap='hot', interpolation='nearest')  # map_data is already a numpy array
    plt.colorbar()
    plt.title(title)
    plt.show()


# Choose a '6' from the dataset
for images, labels in train_loader:
    # Find an image labeled as '6'
    index = (labels == 6).nonzero(as_tuple=True)[0]
    if index.numel() > 0:
        six_image = images[index[0]]
        break

# Generate the occlusion maps
prob_map_6, max_prob_map, label_map = occlusion_test(model, six_image, 6)

# Display the mapsplot_map(prob_map_6.detach().numpy(), "Probability of '6'")
plot_map(prob_map_6.detach().numpy(), "Probability of '6'")
plot_map(max_prob_map.detach().numpy(), "Highest Probability Among All Classes")
plot_map(label_map.detach().numpy(), "Predicted Class Label")
