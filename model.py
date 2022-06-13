import torch
import torchvision.models as models 
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self,x):
        return x

class MyModel(nn.Module):
    def __init__(self, num_class):
        super(MyModel,self).__init__()

        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = Identity()
        self.fc = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, num_class))

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc(x)
        return x 


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_net = nn.Sequential(
            self.CnnBlock(3 , 32),
            self.CnnBlock(32, 32), 
            nn.MaxPool2d((2,2)),
            self.CnnBlock(32, 64),
            self.CnnBlock(64, 64), 
            nn.MaxPool2d((2,2)), 
            self.CnnBlock(64, 128),
            self.CnnBlock(128, 128), 
            nn.MaxPool2d((2,2)), 
            nn.Flatten(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_net(x)
        return self.fc(x)


    def CnnBlock(self, input_dim, output_dim, stride=1, padding = 1):
        conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=(3,3), stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()
        )
        return conv


class CNNModel_Small(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_net = nn.Sequential(
            self.CnnBlock(3 , 32),
            self.CnnBlock(32, 32), 
            nn.MaxPool2d((2,2)),
            self.CnnBlock(32, 64, stride=2),
            self.CnnBlock(64, 64, stride=2), 
            nn.MaxPool2d((2,2)), 
            nn.Flatten(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_net(x)
        return self.fc(x)


    def CnnBlock(self, input_dim, output_dim, stride=1, padding = 1):
        conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=(3,3), stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()
        )
        return conv

