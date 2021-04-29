import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torch

class LeNet5(nn.Module):
    def __init__(self, config):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=(5, 5))  # 5x5+1 params
        self.subsampling = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=(5, 5))  # 5x5+1 params
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=(5, 5))  # 5x5+1 params
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, config['num_classes'])
        
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.optim = optim.SGD(params=self.parameters(),
                               momentum=config['momentum'], lr=config['lr'], nesterov=True, weight_decay=1e-4)
        self.loss=nn.CrossEntropyLoss()
        # self.scheduler=optim.lr_scheduler.StepLR(self.optim,step_size=15,gamma=0.1)
        self.scheduler=optim.lr_scheduler.ExponentialLR(self.optim,gamma=0.98)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.subsampling(x)
        x = F.relu(self.conv2(x))
        x = self.subsampling(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        # print(self.fc1.weight.size())
        # print(torch.nonzero(self.fc1.weight).size(),'weight')
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
    
    def extract_feature(self,x):
        x = F.relu(self.conv1(x))
        x = self.subsampling(x)
        x = F.relu(self.conv2(x))
        x = self.subsampling(x)
        feature = F.relu(self.conv3(x))
        x = feature.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x,feature
