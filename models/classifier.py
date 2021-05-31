import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    '''
    Basic classifier x->y.
    '''
    def __init__(self, net, lr=0.001, device='cuda', criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.device = device
        self.net = net.to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=0)
        self.criterion = criterion
        self.scheduler = None

    def epoch_end(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def forward(self, batch):
        return self.net(batch)

    def train_step(self, batch, labels):
        self.optim.zero_grad()
        output = self.net(batch)
        loss = self.criterion(output, labels)
        loss.backward()
        self.optim.step()
        return {'loss': loss.data}


class ChainedClassifier(nn.Module):
    '''
    Classifier z->y created by chaining pretrained Decoder and Classifier x->y.
    '''
    def __init__(self, generator, classifier):
        super().__init__()
        self.generator = generator
        self.classifier = classifier

    def forward(self, noise):
        imgs = self.generator.decode(noise)
        output = self.classifier(imgs)
        return output
