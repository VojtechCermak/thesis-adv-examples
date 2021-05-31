import torch
import torch.nn as nn
import torch.nn.functional as F
from models.classifier import Classifier

class ConvNet(nn.Module):
    def __init__(self, ch=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(ch, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.main(x)


params = {
    'device': 'cuda',
    'lr': 0.001,
    'net': ConvNet(),
    }

model = Classifier(**params)
model.scheduler = torch.optim.lr_scheduler.StepLR(model.optim, step_size=2, gamma=0.5)