import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ### Feature 1
        # in channels: 3 (xyz)
        # out channels: 10 (10 filters)
        # kernel size: (1, 3) (over width, on each finger)
        self.f1_conv1 = nn.Conv2d( 3, 6, (1, 3) ).cuda()
        self.f1_conv2 = nn.Conv2d( 6, 12, (1, 3) ).cuda()

        self.f1_fc1 = nn.Sequential(
		    nn.Flatten(),
		    nn.Linear(12*5*26, 12)
		).cuda()

        self.f1_fc2 = nn.Linear(12, 6).cuda()

        ### Feature 2
        # in channels: 3 (xyz)
        # out channels: 10 (10 filters)
        # kernel size: (1, 3) (over width, on each finger)
        self.f2_conv1 = nn.Conv2d( 3, 6, (1, 3) ).cuda()
        self.f2_conv2 = nn.Conv2d( 6, 12, (1, 3) ).cuda()

        self.f2_fc1 = nn.Sequential(
		    nn.Flatten(),
		    nn.Linear(12*5*26, 12)
		).cuda()

        self.f2_fc2 = nn.Linear(12, 6).cuda()

        ### Feature 3
        # in channels: 3 (xyz)
        # out channels: 10 (10 filters)
        # kernel size: (1, 3) (over width, on each finger)
        self.f3_conv1 = nn.Conv2d( 3, 6, (1, 3) ).cuda()
        self.f3_conv2 = nn.Conv2d( 6, 12, (1, 3) ).cuda()

        self.f3_fc1 = nn.Sequential(
		    nn.Flatten(),
		    nn.Linear(12*5*26, 12)
		).cuda()

        self.f3_fc2 = nn.Linear(12, 6).cuda()

        # Last one
        self.last_fc = nn.Linear(6*3, 1).cuda() # 3 features, each with 6


    def forward(self, features):
        feature1, feature2, feature3 = features[...,0], features[...,1], features[...,2]
        
        ### Feature 1
        # convolutional layers
        # conv1 - relu
        feature1 = self.f1_conv1(feature1)
        feature1 = F.relu(feature1)

        # conv2 - relu
        feature1 = self.f1_conv2(feature1)
        feature1 = F.relu(feature1)

        # fully connected layers
        # fc1 - relu
        feature1 = self.f1_fc1(feature1)
        feature1 = F.relu(feature1)
        
        # fc2 - relu
        feature1 = self.f1_fc2(feature1)
        feature1 = F.relu(feature1)

        ### Feature 2
        # convolutional layers
        # conv1 - relu
        feature2 = self.f2_conv1(feature2)
        feature2 = F.relu(feature2)

        # conv2 - relu
        feature2 = self.f2_conv2(feature2)
        feature2 = F.relu(feature2)
        
        # fully connected layers
        # fc1 - relu
        feature2 = self.f2_fc1(feature2)
        feature2 = F.relu(feature2)
        
        # fc2 - relu
        feature2 = self.f2_fc2(feature2)
        feature2 = F.relu(feature2)

        ### Feature 3
        # convolutional layers
        # conv1 - relu 
        feature3 = self.f3_conv1(feature3)
        feature3 = F.relu(feature3)

        # conv2 - relu
        feature3 = self.f3_conv2(feature3)
        feature3 = F.relu(feature3)
        
        # fully connected layers
        # fc1 - relu
        feature3 = self.f3_fc1(feature3)
        feature3 = F.relu(feature3)
        
        # fc2 - relu
        feature3 = self.f3_fc2(feature3)
        feature3 = F.relu(feature3)

        # Final
        concat = torch.cat((feature1, feature2, feature3), 1)
        x = self.last_fc(concat)
        x = torch.sigmoid(x)

        return x