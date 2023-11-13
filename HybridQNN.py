import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from Quanv2d import Quanv2d
from tqdm import tqdm

class HybridQNN(nn.Module):
    def __init__(self):
        super(HybridQNN, self).__init__()
        #build a full classical convolutional layer
        self.conv1 = nn.Conv2d(1, 1, 3)
        self.bn1 = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = Quanv2d(1, 2, 2, 3,kernel_size=4,stride=3)
        self.bn2 = nn.BatchNorm2d(2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sigmoid(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

# Define the training function

def train(device, train_loader, optimizer, criterion):
    global model
    model.train()
    train_loss = 0
    correct = 0
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # print(data.shape)
        output = model(data)
        # print(output.shape)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        pbar.set_description(desc=f'Train Loss={train_loss} Batch_id={batch_idx} Accuracy={correct / len(train_loader.dataset):.2f}')
    train_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    return train_loss, accuracy

# Define the test function
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the batch size and number of epochs
batch_size = 25
epochs = 100

# Load the MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))

#load only 50 data
legnth = 100
train_dataset.data = train_dataset.data[:legnth]
train_dataset.targets = train_dataset.targets[:legnth]
test_dataset.data = test_dataset.data[:legnth]
test_dataset.targets = test_dataset.targets[:legnth]

# Create the data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(f'length of dataset :{len(train_loader.dataset)}')
# Initialize the model and move it to the device
model = HybridQNN().to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
print(model)
print(model.parameters())
# Train the model
for epoch in range(1, epochs + 1):
    print(f'epoch : {epoch}')
    train(device, train_loader, optimizer, criterion)
    test_loss, accuracy = test(model, device, test_loader, criterion)
    print('Epoch: {} Test Loss: {:.4f} Accuracy: {:.2f}%'.format(epoch, test_loss, accuracy))





