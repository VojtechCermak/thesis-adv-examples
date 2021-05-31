import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dcgan import DCGAN

class Generator(nn.Module):
    def __init__(self, ch=3, nz=100, nf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, nf*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nf*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nf, ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, ch=3, nf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(ch, nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*8, 1, kernel_size=2, stride=2, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

params = {
    'device': 'cuda',
    'size_z': 100,
    'lr_g': 0.0002, 
    'lr_d': 0.0002,
    'g': Generator(),
    'd': Discriminator(),
    }

model = DCGAN(**params)
model.scheduler_g = torch.optim.lr_scheduler.StepLR(model.optim_g, step_size=25, gamma=0.5)
model.scheduler_d = torch.optim.lr_scheduler.StepLR(model.optim_d, step_size=25, gamma=0.5)