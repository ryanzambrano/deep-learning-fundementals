import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LocallyConnected1d(nn.Module):
    """
    A simple 1D locally connected layer in PyTorch.
    Each input channel is connected to each output channel with distinct weights (no weight sharing).
    """
    def __init__(self, input_channels, output_channels, input_length, kernel_size):
        super(LocallyConnected1d, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.input_length = input_length
        self.output_length = input_length - kernel_size + 1
        
        # Define weights and biases for the layer
        self.weights = nn.Parameter(torch.randn(output_channels, input_channels, self.output_length, kernel_size))
        self.biases = nn.Parameter(torch.randn(output_channels, self.output_length))
        
        
        self.reset_parameters()

    def reset_parameters(self, init_method='effective'):
        if init_method == 'very_slow':
            nn.init.uniform_(self.weights, a=-1e-4, b=1e-4)
        elif init_method == 'effective':
            nn.init.kaiming_uniform_(self.weights, mode='fan_in', nonlinearity='relu')
        elif init_method == 'too_fast':
            nn.init.uniform_(self.weights, a=-0.5, b=0.5)
        nn.init.constant_(self.biases, 0)
        
    def forward(self, x):
        batch_size, _, _ = x.size()
        outputs = torch.zeros((batch_size, self.output_channels, self.output_length), device=x.device)
    
    # Iterate through each position of the output_length
        for i in range(self.output_length):
        # Extract patches across all examples in the batch for this particular window
            patch = x[:, :, i:i+self.kernel_size].unsqueeze(1)  # Shape: [batch_size, 1, input_channels, kernel_size]
        
        # Weights shape transformation for broadcasting: [output_channels, input_channels, kernel_size] -> [1, output_channels, input_channels, kernel_size]
            weighted_sum = torch.sum(patch * self.weights[:, :, i, :].unsqueeze(0), dim=-1) + self.biases[:, i].unsqueeze(0).unsqueeze(-1)
        
        # After summing over the kernel_size dimension, we're left with [batch_size, output_channels, input_channels], we sum over input_channels too.
            outputs[:, :, i] = weighted_sum.sum(dim=2)
        
        return outputs

    
class MultiLayerLocallyConnectedNN(nn.Module):
    
    def __init__(self, init_method='effective'):
        super(MultiLayerLocallyConnectedNN, self).__init__()
        # the three locally conected layers
        self.lc1 = LocallyConnected1d(input_channels=1, output_channels=2, input_length=256, kernel_size=5)
        self.lc2 = LocallyConnected1d(input_channels=2, output_channels=2, input_length=252, kernel_size=5)
        self.lc3 = LocallyConnected1d(input_channels=2, output_channels=2, input_length=248, kernel_size=5)
        
        # Fully connected layer to end
        self.fc = nn.Linear(2 * 244, 10) 

        self.apply_init(init_method)

    def apply_init(self, init_method):
        for m in self.modules():
            if isinstance(m, LocallyConnected1d):
                m.reset_parameters(init_method)
            if isinstance(m, nn.Linear):
                if init_method == 'very_slow':
                    nn.init.uniform_(m.weight, a=-1e-4, b=1e-4)
                elif init_method == 'effective':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif init_method == 'too_fast':
                    nn.init.uniform_(m.weight, a=-0.5, b=0.5)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.lc1(x))  
        x = self.lc2(x)  
        x = torch.sigmoid(self.lc3(x))  # Applying Sigmoid activation on the output of the third locally connected layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  
        return x

    
class HandwrittenDigitDataset(Dataset):
    def __init__(self, file_path):
        # Load the dataset
        data = np.loadtxt(file_path)
        
        # Split into features and labels
        self.labels = torch.LongTensor(data[:, 0])
        self.features = torch.FloatTensor(data[:, 1:]).reshape(-1, 1, 16, 16)  # Reshape to 1x16x16 images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Update dataset to reshape inputs
class UpdatedHandwrittenDigitDataset(HandwrittenDigitDataset):
    def __getitem__(self, idx):
        # Reshape to (1, 256) to match LocallyConnected1d input
        feature = self.features[idx].view(1, -1)  # Reshape
        label = self.labels[idx]
        return feature, label

train_dataset = UpdatedHandwrittenDigitDataset('train.txt')
test_dataset = UpdatedHandwrittenDigitDataset('test.txt')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


init_methods = ['very_slow', 'effective', 'too_fast']







def train_and_validate(model, init_method, train_loader, test_loader, epochs, learning_rate):
    print(f"\nTraining with {init_method} initialization and learning rate {learning_rate}:")


    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:  # Print loss every 10 batches
                print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item()}")

    # Validation 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy}%')


init_methods = ['very_slow', 'effective', 'too_fast']
learning_rates = {'slow': 0.1, 'medium': 0.01, 'fast': 0.001}

for init_method in init_methods:
    for lr_name, lr_value in learning_rates.items():
        print(f"Training with init method: {init_method} and learning rate ({lr_name}): {lr_value}")
        model = MultiLayerLocallyConnectedNN(init_method=init_method)
        train_and_validate(model, init_method, train_loader, test_loader, epochs=3, learning_rate=lr_value)

# Train models with different initialization methods
print("starting training for the ensemble")
models = []
for init_method in init_methods:
    model = MultiLayerLocallyConnectedNN(init_method=init_method)
    train_and_validate(model, init_method, train_loader, test_loader, epochs=3, learning_rate=0.01)
    models.append(model)



class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models

    def forward(self, x):
        # Get predictions
        predictions = [model(x) for model in self.models]
        
        # Average predictions
        avg_predictions = torch.mean(torch.stack(predictions), dim=0)
        
        return avg_predictions

ensemble = Ensemble(models=models)

# Evaluate 
ensemble.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = ensemble(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Ensemble Accuracy: {accuracy}%')
