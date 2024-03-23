import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np



class HandwrittenDigitDataset(Dataset):
    def __init__(self, file_path):
        # Load the dataset
        data = np.loadtxt(file_path)
        
        # Split into features and labels
        self.labels = torch.LongTensor(data[:, 0].astype(int))  # Use int directly here
        self.features = torch.FloatTensor(data[:, 1:].reshape(-1, 1, 16, 16))  # Reshape to 1x16x16 images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Assuming 'train.txt' and 'test.txt' are available in your directory
train_dataset = HandwrittenDigitDataset('train.txt')
test_dataset = HandwrittenDigitDataset('test.txt')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)



class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, init_mode='effective'):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # Correcting the input size of the fully connected layer
        self.fc = nn.Linear(128 * 2 * 2, num_classes)

        self.apply(self._init_weights(init_mode))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.sigmoid(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

    
    def _init_weights(self, mode):
        def init_func(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if mode == 'very_slow':
                    nn.init.uniform_(m.weight, a=-1e-4, b=1e-4)
                elif mode == 'effective':
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                elif mode == 'too_fast':
                    nn.init.uniform_(m.weight, a=-0.5, b=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        return init_func

def train_and_validate(model, train_loader, test_loader, epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training
    model.train()
    for epoch in range(epochs):
        print("epoch:" + str(epoch))
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100. * correct / total
    print(f'Accuracy: {accuracy}%')

# Initialize and train models with different initializations
init_modes = ['very_slow', 'effective', 'too_fast']
learning_rates = [0.1, 0.001, 0.00001]  # Define slow, medium, and fast learning rates
epochs = 3


models = []
for mode in init_modes:
    for lr in learning_rates:
        print(f"\nTraining with {mode} initialization and learning rate {lr}:")
        model = SimpleCNN(init_mode=mode)
        train_and_validate(model, train_loader, test_loader, epochs, lr)
        if lr == 0.001:
            models.append(model)
        

class SimpleCNNBN(nn.Module):
    def __init__(self, num_classes=10, init_mode='effective'):
        super(SimpleCNNBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        #Adjust the input size to the fully connected layer based on your network's architecture
        # For a 16x16 input image that's pooled three times, the size is reduced by a factor of 2^3 = 8
        #So the new size is 2x2 (16/8 = 2)
        self.fc = nn.Linear(128 * 2 * 2, num_classes) 

        self.apply(self._init_weights(init_mode))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.bn3(self.conv3(x)), 2))
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x
    def _init_weights(self, mode):
        def init_func(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if mode == 'effective':
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                elif mode == 'very_slow':
                    nn.init.uniform_(m.weight, a=-1e-4, b=1e-4)
                elif mode == 'too_fast':
                    nn.init.uniform_(m.weight, a=-0.5, b=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        return init_func
    
def experiment_momentum(momentum_values, train_loader, test_loader, epochs, learning_rate, best_batch_size):
    results = {}
    for momentum in momentum_values:

        model = SimpleCNNBN(init_mode='effective')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = nn.CrossEntropyLoss()
        
        print(f"\nTraining with momentum {momentum}:")
        train_and_validate(model, train_loader, test_loader, epochs, learning_rate)
        
        # Evaluate the model to get the accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100. * correct / total
        print(f'Accuracy with momentum {momentum}: {accuracy}%')
        results[momentum] = accuracy
    return results

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Momentum values to experiment with
momentum_values = [0.5, 0.9, 0.99]


# Define the CNN model
model = SimpleCNNBN(num_classes=10)

# Learning rate and epochs for demonstration
learning_rate = 0.001
epochs = 3

# Train and validate the model with a small batch size
print("\nTraining with small batch size (may be less effective for batch normalization):")
train_loader_small = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader_small = DataLoader(test_dataset, batch_size=4)
train_and_validate(model, train_loader_small, test_loader_small, epochs, learning_rate)


model = SimpleCNNBN(num_classes=10)

#Train and validate the model with a larger batch size
print("\nTraining with larger batch size (more effective for batch normalization):")
train_loader_large = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader_large = DataLoader(test_dataset, batch_size=64)
train_and_validate(model, train_loader_large, test_loader_large, epochs, learning_rate)


results = experiment_momentum(momentum_values, train_loader, test_loader, 3, 0.001, 64)

# the results
print("\nExperiment results:", results)

def ensemble_predict(models, inputs):
    total_preds = torch.zeros(inputs.size(0), 10) 
    for model in models:
        model.eval()  
        with torch.no_grad():  
            preds = F.softmax(model(inputs), dim=1)  # Use softmax to get probabilities
            total_preds += preds
    avg_preds = total_preds / len(models)
    return avg_preds


correct = 0
total = 0
for inputs, labels in test_loader:
    inputs, labels = inputs, labels  
    ensemble_predictions = ensemble_predict(models, inputs)
    _, predicted = torch.max(ensemble_predictions.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Ensemble Accuracy: {accuracy}%')