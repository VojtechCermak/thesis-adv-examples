import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vae import VanillaVAE

class Encoder(nn.Module):
    def __init__(self, ch=1, nz=64, nf=32):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(ch, nf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*8, nz*2, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        mu, logvar = torch.chunk(self.main(x), 2, dim=1)
        return mu, logvar



class Decoder(nn.Module):
    def __init__(self, ch=1, nz=64, nf=32):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, nf*8, kernel_size=4, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(nf*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(nf*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nf, ch, kernel_size=1, stride=1, padding=2, bias=True),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


params = {
    'device': 'cuda',
    'lr': 0.001,
    'loss': 'mse',
    'beta': 3,
    'size_z': 64,
    'encoder': Encoder(),
    'decoder': Decoder(),
    }

model = VanillaVAE(**params)