from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pickle
filtered_classes = ['bear','tiger']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self,ftr_out):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, ftr_out)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x




def demo(model):

    with open('demo12', 'rb') as fo:
        demodict = pickle.load(fo, encoding='bytes')
    torch.no_grad()
    fig,ax = plt.subplots(3, 4,figsize=(14,9))
    t1=time.time()
    correct=0
    correct_labels=[0]*6+[1]*6
    for i, origdata in enumerate(demodict[b'data']):
        origdata=origdata.reshape((3,32*32)).transpose()
        origdata=origdata.reshape(32,32,3)

######### BELOW:preprocess the origdata for model input and predict using the model #######
######### modify code in this block to predict ######################################
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(28, antialias=False),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        data=transform(origdata)
        data=data.reshape((1,3,28,28))
        output=model(data)
        pred = output.argmax(dim=1, keepdim=True)
        print(type(pred),pred.item(),correct_labels[i])
        if pred.item()==correct_labels[i]:correct+=1
###############################################################################################
###########################  Do not change the code below #####################################
        ax[i//4][i%4].axis('off')
        ax[i//4][i%4].set_title(f'predicted: {filtered_classes[pred]}')
        ax[i//4][i%4].imshow(origdata)
    t2=time.time()
    fig.suptitle('time taken={:6f} sec. Correct images {}'.format(t2-t1,correct),fontsize=16)
    plt.savefig('ex.png')
    plt.ioff()
    plt.show()
if __name__ == '__main__':
    ######### BELOW: load your model ###########
    model = Net(len(filtered_classes))
    model.load_state_dict(torch.load('cifar_cnn.pt'))
    model.eval()
    ####################################################
    demo(model)

