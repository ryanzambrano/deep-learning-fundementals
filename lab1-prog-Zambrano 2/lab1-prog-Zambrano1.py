import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Define the fully connected neural network
class FullyConnectedNN(nn.Module):
    def __init__(self, init_mode='effective'):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

        # Initialize parameters as per the mode specified
        self.initialize_parameters(init_mode)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def initialize_parameters(self, mode):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if mode == 'slow':
                    nn.init.uniform_(m.weight, a=-0.0001, b=0.0001)
                    nn.init.constant_(m.bias, 0)
                elif mode == 'effective':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
                elif mode == 'fast':
                    nn.init.normal_(m.weight, mean=0, std=1)
                    nn.init.constant_(m.bias, 0)

# Create models with different initialization modes
model_slow = FullyConnectedNN(init_mode='slow')
model_effective = FullyConnectedNN(init_mode='effective')
model_fast = FullyConnectedNN(init_mode='fast')
# Function to load data
def load_data(filepath):
    data = np.loadtxt(filepath)
    labels = torch.from_numpy(data[:, 0]).long()
    features = torch.from_numpy(data[:, 1:]).float()
    return TensorDataset(features, labels)

# Prepare DataLoader
def get_data_loader(dataset, batch_size=64):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load datasets
train_dataset = load_data('train.txt')
test_dataset = load_data('test.txt')

train_loader = get_data_loader(train_dataset)
test_loader = get_data_loader(test_dataset)

# Initialize the model, loss function, and optimizer
model = FullyConnectedNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_and_validate(model, train_loader, test_loader, criterion, learning_rate, epochs=10):
    # Initialize the optimizer with the specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on the test set with LR {learning_rate}: {100 * correct / total}%')

# Load datasets
train_dataset = load_data('train.txt')  # Adjust path if necessary
test_dataset = load_data('test.txt')    # Adjust path if necessary

train_loader = get_data_loader(train_dataset)
test_loader = get_data_loader(test_dataset)

# Train and validate each model with its specific initialization strategy
init_modes = ['slow', 'effective', 'fast']
learning_rates = {'very_slow': 1e-5, 'effective': 0.001, 'too_fast': 0.1}

for init_mode in init_modes:
    for lr_name, lr_value in learning_rates.items():
        print(f"\nTraining with {init_mode} initialization and {lr_name} learning rate: {lr_value}")
        model_lr = FullyConnectedNN(init_mode=init_mode)  # Initialize model with the current mode
        train_and_validate(model_lr, train_loader, test_loader, criterion, lr_value, epochs=10)

        
models = [FullyConnectedNN(init_mode='effective') for _ in range(3)]
print("Starting ensmble training")
for i, model in enumerate(models):
    print(f"Training model {i+1}")
    train_and_validate(model, train_loader, test_loader, criterion, 0.001, epochs=10)

def ensemble_predict(models, data_loader):
    total_preds = []
    total_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs, labels
            preds = [F.softmax(model(inputs), dim=1) for model in models]
            # Average predictions across models
            avg_preds = torch.mean(torch.stack(preds), dim=0)
            _, predicted_labels = torch.max(avg_preds, dim=1)
            total_preds.extend(predicted_labels.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
    return total_preds, total_labels

# Calculate ensemble accuracy
def calculate_ensemble_accuracy(models, data_loader):
    predictions, labels = ensemble_predict(models, data_loader)
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = 100.0 * correct / len(labels)
    return accuracy

ensemble_acc = calculate_ensemble_accuracy(models, test_loader)
print(f'Ensemble Accuracy: {ensemble_acc:.2f}%')


# model with dropout
class FullyConnectedNNWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(FullyConnectedNNWithDropout, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
# Moderate dropout rate
model_effective_dropout = FullyConnectedNNWithDropout(dropout_rate=0.5)
# High dropout rate
model_ineffective_dropout = FullyConnectedNNWithDropout(dropout_rate=0.9)
print("Training model with effective dropout rate (0.5):")
train_and_validate(model_effective_dropout, train_loader, test_loader, criterion, 0.001, epochs=10)

print("\nTraining model with ineffective dropout rate (0.9):")
train_and_validate(model_ineffective_dropout, train_loader, test_loader, criterion, 0.001, epochs=10)



# Define evaluation function
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# fgsm adversarial examples
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Training function with adversarial training
def train_with_adversarial(model, train_loader, epsilon, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data, target
            data.requires_grad = True
            output = model(data)
            init_loss = F.cross_entropy(output, target)

            model.zero_grad()
            init_loss.backward()
            data_grad = data.grad.data
            perturbed_data = fgsm_attack(data, epsilon, data_grad)

            # Combine clean and adversarial examples
            combined_data = torch.cat([data, perturbed_data], dim=0)
            combined_target = torch.cat([target, target], dim=0)

            optimizer.zero_grad()
            combined_output = model(combined_data)
            loss = F.cross_entropy(combined_output, combined_target)
            loss.backward()
            optimizer.step()
        
        # Evaluate on clean data
        clean_accuracy = evaluate(model, test_loader)
        print(f"Clean Test Accuracy after epoch {epoch+1}: {clean_accuracy}%")


model = FullyConnectedNN()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

epsilon = 0.2  #magnitude
train_with_adversarial(model, train_loader, epsilon, epochs=10)

# Evaluate the final model on clean data
clean_accuracy = evaluate(model, test_loader)
print(f"Final Clean Test Accuracy: {clean_accuracy}%")