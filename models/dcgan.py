import torch
import torch.nn as nn
import torch.nn.functional as F

class DCGAN(nn.Module):
    '''
    DCGAN model based on Pytorch DCGAN tutorial.
    Modules: Generator (z -> x), Distriminator (x, x_rec -> {0,1})
    '''
    def __init__(self, device, g, d, size_z, lr_g=0.0002, lr_d=0.0002, smoothing=0.0):
        super().__init__()
        self.device = device
        self.g = g.to(device)
        self.d = d.to(device)
        self.size_z = size_z
        self.smoothing = smoothing
        self.optim_g = torch.optim.Adam(self.g.parameters(), lr=lr_g, betas=(0.5, 0.999), weight_decay=0)
        self.optim_d = torch.optim.Adam(self.d.parameters(), lr=lr_d, betas=(0.5, 0.999), weight_decay=0)
        self.criterion = nn.BCELoss()

        self.scheduler_g = None
        self.scheduler_d = None
        self.apply(self.weights_init)

    def epoch_end(self):
        if self.scheduler_g is not None:
            self.scheduler_g.step()
        if self.scheduler_d is not None:
            self.scheduler_d.step()

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def decode(self, batch):
        '''
        Decodes batch of random normal vector to image space.
        Tensor dimensions: (B, Z-size, 1, 1) -> (B, C, H, W)
        '''
        return self.g(batch)

    def train_step(self, batch):
        batch = batch.to(self.device)
        noise = torch.randn(batch.shape[0], self.size_z, 1, 1, device=self.device)
        label_fake = torch.full((batch.shape[0],), 0.0 + self.smoothing, device=self.device)
        label_real = torch.full((batch.shape[0],), 1.0 + self.smoothing, device=self.device)
        
        # 1. Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        self.d.zero_grad()

        # 1.1 Train with all-real batch
        output = self.d(batch)
        error_real = self.criterion(output, label_real)
        error_real.backward()

        # 1.2 Train with all-fake batch
        fake = self.g(noise)
        output = self.d(fake.detach())
        error_fake = self.criterion(output, label_fake)
        error_fake.backward()

        # 1.3 Update discriminator
        error_d = error_real + error_fake
        self.optim_d.step()

        # 2. Update G network: maximize log(D(G(z)))
        self.g.zero_grad()
        output = self.d(fake)
        error_g = self.criterion(output, label_real)
        error_g.backward()
        self.optim_g.step()

        return {'loss_d': error_d.data, 'loss_g': error_g.data}
