import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from Quanv2d import Quanv2d
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import random
import itertools
import numpy as np
from typing import Union, List, Iterator
from HybridQNN import HybridQNN
from HybridQNN_Transfer import HybridQNN_T


class HybridQNN_Multi(nn.Module):
    '''
    A hybrid quantum convolutional neural network
    constucted by a classical convolutional layer and a quantum convolutional layer
    '''
    def __init__(self):
        super(HybridQNN_Multi, self).__init__()
        #build a full classical convolutional layer
        '''
        write the layer needed for the model
        '''
        self.HybridQNN1 = HybridQNN()
        self.HybridQNN2 = HybridQNN()
        self.HybridQNN3 = HybridQNN_T()
        self.Linear1 = nn.Linear(30, 10)

        # self.linear2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        '''
        forward function for the model
        args
            x: input tensor
        return
            x: output tensor
        '''
        '''
        write the forward function for the model
        default as 
        HybridQNN(
            (conv1): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1))
            (bn1): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
            (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (conv2): Quanv2d(
                (qnn): TorchConnector()
            )
            (bn2): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU()
            (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (flatten): Flatten(start_dim=1, end_dim=-1)
            (linear): Linear(in_features=32, out_features=10, bias=True)
        )
        '''
        with torch.no_grad():
            x1 = self.HybridQNN1(x)
            x2 = self.HybridQNN2(x)
            x3 = self.HybridQNN3(x)
        x = torch.cat((x1,x2,x3),dim=1)
        x = self.Linear1(x)
        return x

# Define the training function

def train(
        model:nn.modules,
        device:torch.device, 
        train_loader: DataLoader, 
        optimizer:optim.Optimizer, 
        criterion: nn.Module
        )->tuple[float, float, nn.Module]:
    '''
    basic train function generate by github-copilot
    args
        device: device to train the model
        train_loader: training data loader
        optimizer: optimizer for the model
        criterion: loss function    
    return
        train_loss: loss of the training process
        accuracy: accuracy of the training process
        model: trained model
    '''
    model.train()
    train_loss = 0
    correct = 0
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        pbar.set_description(desc=f'Train Loss={train_loss/len(train_loader.dataset ):.4f}'+
                                  f'|Batch_id={batch_idx}'+
                                  f'|Accuracy={correct / len(train_loader.dataset):.2f}')
    # train_loss /= len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    return train_loss, accuracy, model  

# Define the test function
def test(
        model:nn.modules, 
        device:torch.device, 
        test_loader:DataLoader, 
        criterion:nn.Module
        )->tuple[float, float]:
    '''
    basic test function generate by github-copilot
    args
        device: device to train the model
        test_loader: test data loader
        criterion: loss function
    return
        test_loss: loss of the test process
        accuracy: accuracy of the test process
    '''
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

def Confusion_Matrix(model : nn.Module,
                    device : torch.device,
                    test_loader : DataLoader) -> torch.Tensor:
        '''
        generate the confusion matrix of the model
        args
            model: trained model
            device: device to train the model
            test_loader: test data loader
        return
            confusion_matrix: confusion matrix of the model
        '''
        model.eval()
        confusion_matrix = torch.zeros(10,10)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).argmax(dim=1, keepdim=True)
                for i in range(len(target)):
                    confusion_matrix[target[i]][output[i]] += 1
        return confusion_matrix

def plot_confusion_matrix(cm :Union[np.ndarray,torch.Tensor], 
                          classes : Iterator, 
                          normalize : bool = False, 
                          title : str = 'Confusion matrix', 
                          cmap : plt.cm.ColormapRegistry = plt.cm.Blues
                          )->None:
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    args
        cm: confusion matrix
        classes: classes of the confusion matrix
        normalize: normalize the confusion matrix or not
        title: title of the confusion matrix
        cmap: color map of the confusion matrix
    return
        None
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def Train_Hybrid_QNN(Net : nn.Module,
                     optimizer : optim.Optimizer,
                     criterion : nn.Module,
                     train_dataset : datasets,
                     test_dataset : datasets,
                     **kwargs) -> None:
    '''
    Train a HybridQNN model
    args
        Net: HybridQNN
        optimizer: optimizer for the model
        criterion: loss function
        train_dataset: training dataset
        test_dataset: test dataset
        kwargs: hyperparameters
    return
        None
    Save
        model: trained model (to data/{model_name}/{model_path})
        results: results of the training process (to data/{model_name}/results.pt)
        accuracy.png: plot of the accuracy (to data/{model_name}/accuracy.png)
        loss.png: plot of the loss (to data/{model_name}/loss.png)
    '''
    defaule_kwargs = {'legnth':500,
                      'batch_size':50,
                      'epochs':10,
                      'model_name':'HybridQNN',
                      'model_path':'model.pt',
                      'learning_rate':0.01,
                      'mode':'new_model',
                      'seed':0}
    
    for key in kwargs:
        if key in defaule_kwargs:
            defaule_kwargs[key] = kwargs[key]
        else:
            raise ValueError(f'key {key} not in defaule_kwargs')
        
    legnth = defaule_kwargs['legnth']
    batch_size = defaule_kwargs['batch_size']
    epochs = defaule_kwargs['epochs']
    model_name = defaule_kwargs['model_name']
    model_path = defaule_kwargs['model_path']
    learning_rate = defaule_kwargs['learning_rate']
    mode = defaule_kwargs['mode']

    torch.manual_seed(seed)
    #make directory
    os.makedirs(f'data/{model_name}',exist_ok=True)
    #load data
    train_dataset.data = train_dataset.data[:legnth]
    train_dataset.targets = train_dataset.targets[:legnth]
    test_dataset.data = test_dataset.data[:int(legnth/2)]
    test_dataset.targets = test_dataset.targets[:int(legnth/2)]

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the model and move it to the device

    model = Net.to(device)
    if mode == 'old_model':
        model.load_state_dict(torch.load(f'data/{model_name}/{model_path}'))
        #load results
        results = torch.load(f'data/{model_name}/results.pt')
    else:
        results = {'train_loss':[],'train_accu':[],'test_loss':[],'test_accu':[],'best_loss':1e5}
        #check if the model exists
        if os.path.exists(f'data/{model_name}/{model_path}'):
            check = input('model exists, press enter to overwrite(y/n)')
            if check == 'n' or check == 'N':
                build = input('build a new model?(y/n)')
                if build == 'n' or build == 'N':
                    raise ValueError('model exists')
                elif build == 'y' or build == 'Y':
                    model_name = f'{model_name}_{random.randint(0,100)}'
                    os.makedirs(f'data/{model_name}',exist_ok=True)
                    print(f'new model name: {model_name}')
                else:
                    raise ValueError('invalid input')
            elif check == 'y' or check == 'Y':
                pass
            else:
                raise ValueError('invalid input')

    print(model)

    # Define the optimizer and loss function
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # Train the model
    if epochs > 0:
        for epoch in range(1, epochs + 1):
            print(f'epoch : {epoch}')
            train_loss, train_accu, model = train(model,device, train_loader, optimizer, criterion)
            test_loss , accuracy           = test(model, device, test_loader, criterion)
            results['train_loss'].append(train_loss)
            results['train_accu'].append(train_accu)
            results['test_loss'].append(test_loss)
            results['test_accu'].append(accuracy)
            if test_loss < results['best_loss']:
                results['best_loss'] = test_loss
                #save the model for future use
                torch.save(model.state_dict(), f'data/{model_name}/{model_path}')
            #save results
            torch.save(results,f'data/{model_name}/results.pt')
            print('Epoch: {} Test Loss: {:.4f} Accuracy: {:.2f}%'.format(epoch, test_loss, accuracy))
    elif epochs == 0:
        pass
    else:
        raise ValueError('invalid epochs')

    CM = Confusion_Matrix(model,device,test_loader)
    plot_confusion_matrix(CM.numpy(),classes=range(CM.shape[0]),normalize=True)
    plt.savefig(f'data/{model_name}/confusion_matrix.png')
    plt.clf()

    #plot the loss and accuracy
    plt.plot(results['train_loss'],label='train_loss')
    plt.plot(results['test_loss'],label='test_loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(f'data/{model_name}/loss.png')
    plt.clf()
    plt.plot(results['train_accu'],label='train_accu')
    plt.plot(results['test_accu'],label='test_accu')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(f'data/{model_name}/accuracy.png')
    plt.clf()


if __name__ == '__main__':
    #some hyperparameters
    legnth = 500
    batch_size = 50
    epochs = 10
    model_name = 'HybridQNN_Multi'
    model_path = 'model.pt'
    learning_rate = 0.01
    mode = 'new_model'
    seed = 0
    # Load the MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    optimizer = optim.Adam
    criterion = nn.CrossEntropyLoss()
    Net = HybridQNN_Multi()
    Net.HybridQNN1.load_state_dict(torch.load('data/HybridQNN/model.pt'))
    Net.HybridQNN2.load_state_dict(torch.load('data/HybridQNN_mac/model.pt'))
    Net.HybridQNN3.load_state_dict(torch.load('data/HybridQNN_T/model.pt'))
    Train_Hybrid_QNN(Net,optimizer,criterion,train_dataset,test_dataset,
                     legnth=legnth,batch_size=batch_size,epochs=epochs,
                     model_name=model_name,model_path=model_path,learning_rate=learning_rate,mode=mode,seed=seed)