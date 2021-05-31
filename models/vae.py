import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

class VanillaVAE(nn.Module):
    '''
    Vanilla Variational Autoencoder based on original Kingsma paper.
    '''
    def __init__(self, encoder, decoder, device, size_z, lr=1e-3, loss='bce', beta=1):
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        self.size_z = size_z
        self.loss = loss
        self.beta = beta
        self.optimizer = torch.optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=lr)
        self.scheduler = None

    def epoch_end(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def loss_function(self, img_rec, img, mu, logvar):
        """
        Loss =  KL-Divergence + Reconstruction loss (BCE)
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        if self.loss == 'bce':
            reconstruction_loss = F.binary_cross_entropy(img_rec, img, reduction='sum')
        elif self.loss == 'mse':
            reconstruction_loss = F.mse_loss(img_rec, img, reduction='sum')

        kl_divergence = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        loss = reconstruction_loss + self.beta*kl_divergence
        return loss

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

        mu, logvar = self.encoder(batch)
        return mu

    def decode(self, batch):
        '''
        Decodes batch of random normal vector to image space.
        Tensor dimensions: (B, Z-size, 1, 1) -> (B, C, H, W)
        '''

        return self.decoder(batch)

    def train_step(self, batch):
        '''
        Defines model training step for each batch.
        '''

        self.optimizer.zero_grad()
        mu, logvar = self.encoder(batch)
        z_enc = self.reparametrize(mu, logvar)
        img_rec = self.decoder(z_enc)

        loss = self.loss_function(img_rec, batch, mu, logvar)
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.data}