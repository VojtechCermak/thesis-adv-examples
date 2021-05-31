import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

class ALI(nn.Module):
    '''
    Adversarially Learned Inference model based on Dumoulin (2016)
    Modules: GeneratorX (z -> x), GeneratorZ(x -> mu, logvar) and joint Distriminator (x, x -> {0,1})
    '''
    def __init__(self, device, gx, gz, d, size_z, lr_g=1e-4, lr_d=1e-4, betas=(.5, 0.999), init_normal=True):
        super().__init__()
        self.device = device
        self.gx = gx.to(device)
        self.gz = gz.to(device)
        self.d = d.to(device)
        self.size_z = size_z
        self.optim_g = torch.optim.Adam(chain(self.gx.parameters(), self.gz.parameters()), lr=lr_g, betas=betas, weight_decay=0)
        self.optim_d = torch.optim.Adam(self.d.parameters(), lr=lr_d, betas=betas, weight_decay=0)

        self.scheduler_g = None
        self.scheduler_d = None
        if init_normal:
            self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def epoch_end(self):
        if self.scheduler_g is not None:
            self.scheduler_g.step()
        if self.scheduler_d is not None:
            self.scheduler_d.step()

    def reparametrize(self, mu, logvar):
        """
        VAE Reparametrization trick to bypass random nodes.
        """

        std = torch.exp(0.5*logvar)
        noise = torch.randn_like(std)
        return mu + (noise * std)

    def encode(self, batch):
        '''
        Encodes batch of images to latent vectors
        Tensor dimensions: (B, C, H, W) -> (B, Z-size, 1, 1)
        '''

        mu, logvar = self.gz(batch)
        return mu

    def decode(self, batch):
        '''
        Decodes batch of random normal vector to image space.
        Tensor dimensions: (B, Z-size, 1, 1) -> (B, C, H, W)
        '''

        return self.gx(batch)

    def train_step(self, batch):
        mu, logvar = self.gz(batch)        
        z_enc = self.reparametrize(mu, logvar)
        z = torch.randn_like(z_enc)

        # Loss
        img_fake = self.gx(z)
        d_true = self.d(x=batch, z=z_enc)
        d_fake = self.d(x=img_fake, z=z)
        loss_d = torch.mean(F.softplus(-d_true) + F.softplus(d_fake))
        loss_g = torch.mean(F.softplus(d_true) + F.softplus(-d_fake))

        # backprop discriminator
        self.d.zero_grad()
        loss_d.backward(retain_graph=True)
        self.optim_d.step()

        # backprop generators
        self.gx.zero_grad()
        self.gz.zero_grad()
        loss_g.backward()
        self.optim_g.step()
        return {'loss_d': loss_d.data, 'loss_g': loss_g.data}