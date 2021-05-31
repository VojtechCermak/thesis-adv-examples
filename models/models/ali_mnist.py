import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ali import ALI

class GeneratorX(nn.Module):
    def __init__(self, ch=1, nz=32, nf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, nf*8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(nf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),

            nn.ConvTranspose2d(nf, ch, kernel_size=1, stride=1, padding=2),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)

class GeneratorZ(nn.Module):
    def __init__(self, nz=32, nf=64, ch=1):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(ch, nf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nf*4, nz*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nz*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nz*2, nz*2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        mu, logvar = torch.chunk(self.main(x), 2, dim=1)
        return mu, logvar

class Discriminator(nn.Module):
    def __init__(self, nz=32, nf=64, ch=1):
        super().__init__()
        self.part_x = nn.Sequential(
            nn.Conv2d(ch, nf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(nf*4, nz*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nz*2),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(p=0.2),
        )
        
        self.part_z = nn.Sequential(
            nn.Conv2d(nz, nz*2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(nz*2, nz*2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(p=0.2),
        )

        self.joint = nn.Sequential(
            nn.Conv2d(nz*4, nz*4, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(nz*4, nz*4, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(nz*4, 1, kernel_size=1, stride=1, padding=0),
            nn.Dropout2d(p=0.2),
        )


    def forward(self, z, x):
        concat = torch.cat((self.part_x(x), self.part_z(z)), dim=1)        
        return self.joint(concat).unsqueeze(2).unsqueeze(3)

params = {
    'device': 'cuda',
    'lr_g': 1e-4,
    'lr_d': 1e-4,
    'betas': (.5, 0.999),
    'size_z': 32,
    'init_normal': False,
    'gx': GeneratorX(),
    'gz': GeneratorZ(),
    'd': Discriminator(),
    }

model = ALI(**params)
model.scheduler_g = torch.optim.lr_scheduler.StepLR(model.optim_g, step_size=10, gamma=0.5)
model.scheduler_d = torch.optim.lr_scheduler.StepLR(model.optim_d, step_size=10, gamma=0.5)