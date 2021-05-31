import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ali import ALI

class GeneratorX(nn.Module):
    def __init__(self, nz=64, ch=3):
        super().__init__()
        self.main =  nn.Sequential(
            nn.ConvTranspose2d(nz, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
    
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
    
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),

            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
    
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
    
            nn.Conv2d(32, ch, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.main(z)
        return (x * 2) - 1

class GeneratorZ(nn.Module):
    def __init__(self, nz=64, ch=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(ch, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
    
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
    
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
    
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
    
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
    
            nn.Conv2d(512, nz*2, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, x):
        x = (x * 0.5) + 0.5
        mu, logvar = torch.chunk(self.main(x), 2, dim=1)
        return mu, logvar

class Discriminator(nn.Module):
    def __init__(self, nz=64, ch=3):
        super().__init__()
        self.part_x = nn.Sequential(
            nn.Conv2d(ch, 32, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(p=0.2),
    
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(p=0.2),
    
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(p=0.2),
    
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(p=0.2),
    
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(p=0.2),
            )
        
        self.part_z = nn.Sequential(
            nn.Conv2d(nz, 512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(p=0.2),
            )
        
        self.joint = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(p=0.2),
    
            nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0),
            nn.Dropout2d(p=0.2),
        )

    def forward(self, z, x):
        x = (x * 0.5) + 0.5
        concat = torch.cat((self.part_x(x), self.part_z(z)), dim=1)        
        return self.joint(concat)

params = {
    'device': 'cuda',
    'lr_g': 1e-4,
    'lr_d': 1e-5,
    'betas': (.5, 1e-3),
    'size_z': 64,
    'init_normal': True,
    'gx': GeneratorX(),
    'gz': GeneratorZ(),
    'd': Discriminator(),
    }

model = ALI(**params)