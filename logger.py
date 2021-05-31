import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np

import os
import json
import shutil
import inspect
from metrics import swd_metric, is_metric
from attacks import fgsm, ifgsm

class Logger():

    def __init__(self, logdir, dataset, task, model, seed=None):
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        self.task = task

        if task == 'end':
            self.summary = EncoderDecoderSummary(dataset, seed=seed, size_z=model.size_z)
        elif task == 'gen':
            self.summary = GenerativeSummary(dataset, seed=seed, size_z=model.size_z)
        elif task == 'cls':
            self.summary = ClassifierSummary(dataset)
        else:
            raise ValueError(f'Invalid task: {task}')   

    def log_summary(self, step, model):
        '''
        Log epoch with model modules in eval mode.
        '''
        inference_mode(self.summary.log_summary, model=model)(step, self.writer, model)


    def print_summary(self, model):
        '''
        Log epoch with model modules in eval mode.
        '''
        inference_mode(self.summary.print_summary, model=model)(model)


    def save_model(self, model):
        '''
        Save model state dict.
        '''
        torch.save(model.state_dict(), self.logdir + f'/model_state_dict.pth')


    def save_source(self, model):
        '''
        Save modules source code for debugging
        '''
        with open(self.logdir + '/source_model.txt', 'w') as text:
            text.write(inspect.getsource(model.__class__))


    def save_dictionary(self, dictionary, filename):
        '''
        Save dictionary as json to specified filename in logdir
        '''
        with open(self.logdir + '/' +  filename, 'w') as f:
            json.dump(dictionary, f, indent=2)

    def save_file(self, file):
        '''
        Copies file to logdir.
        '''
        directory, filename = os.path.split(file)
        shutil.copyfile(file, self.logdir + '/' +  filename)

class GenerativeSummary():

    def __init__(self, dataset, size_z, seed=None):
        '''
        Model shapes:
        Decoder / Generator function: z (B, size_z, 1, 1) -> image (B, C, W, H)
        '''
        if seed is not None:
            torch.manual_seed(seed)
        self.seed = seed
        self.dataset = dataset
        self.size_z = size_z

        # Generate validation data
        self.noise = torch.randn(36, size_z, 1, 1)

        # Fixed images and noise for SWD
        self.swd_noise = torch.randn(1000, size_z, 1, 1)
        self.swd_images = next(iter(DataLoader(dataset, batch_size=1000, shuffle=True)))[0]

        # Fixed noise and images for interpolation grids 
        corners = torch.randn([size_z, 1, 2, 2])
        self.grid_noise = F.interpolate(corners, size=6, mode='bilinear', align_corners=True)

    def swd(self, decoder, device='cuda'):
        return swd_metric(self.swd_images, decoder(self.swd_noise) , device=device)

    def generate(self, decoder):
        return make_grid(decoder(self.noise), nrow=6, normalize=True)


    def interpolate_noise(self, decoder):
        '''
        Returns interpolation of decoded latent space between 4 random latent vectors.
        '''
        images = decoder(self.grid_noise.view(self.size_z, 36, 1, 1).permute(1, 0, 2, 3))
        return make_grid(images, nrow=6, normalize=True)

    def inception_score(self, decoder, classifier, repeat=50, batch_size=1024, device='cuda'):
        '''
        Calculate inception score with given classifier and number of repeats.
        '''
        data = []
        for i in range(repeat):
            z = torch.randn(batch_size, self.size_z, 1, 1, device=device)
            imgs = decoder(z)
            output = F.softmax(classifier(imgs), 1).data.cpu()
            data.append(is_metric(output).item())

        data = np.array(data)
        return np.mean(data), np.std(data)

    @torch.no_grad()
    def log_summary(self, step, writer, model):
        '''
        Log epoch summary to tensorboard writer.
        '''
        decoder = model.decode

        # SWD metric
        swd_metrics = self.swd(decoder)
        for level, value in enumerate(swd_metrics):
            writer.add_scalar(f'swd/swd_level{level}', value, global_step=step)

        # Latent space quality indicators
        writer.add_image('generated', self.generate(decoder), global_step=step)
        writer.add_image('interpolate_noise', self.interpolate_noise(decoder), global_step=step)


class EncoderDecoderSummary(GenerativeSummary):

    def __init__(self, dataset, size_z, seed=None):
        '''
        Encoder/Decoder shapes:
        Decoder function: z (B, size_z, 1, 1) -> image (B, C, W, H)
        Encoder function: image (B, C, W, H) -> z (B, size_z, 1, 1)
        '''
        if seed is not None:
            torch.manual_seed(seed)
        self.seed = seed
        self.dataset = dataset
        self.size_z = size_z

        # Generate validation data
        self.noise = torch.randn(36, size_z, 1, 1)
        self.images = next(iter(DataLoader(dataset, batch_size=12, shuffle=True)))[0]

        # Fixed images and noise for SWD
        self.swd_noise = torch.randn(1000, size_z, 1, 1)
        self.swd_images = next(iter(DataLoader(dataset, batch_size=1000, shuffle=True)))[0]

        # Fixed noise and images for interpolation grids 
        corners = torch.randn([size_z, 1, 2, 2])
        self.grid_noise = F.interpolate(corners, size=6, mode='bilinear', align_corners=True)
        self.grid_images = next(iter(DataLoader(dataset, batch_size=4, shuffle=True)))[0]


    def reconstruct(self, encoder, decoder):
        '''
        Returns concatenated original and reconstructed images and their l2 distance.
        '''
        images_rec = decoder(encoder(self.images))          
        images_compare = torch.cat([self.images, images_rec])
        l2_distance = torch.dist(self.images, images_rec)
        return make_grid(images_compare, nrow=6, normalize=True), l2_distance


    def interpolate_images(self, encoder, decoder):
        '''
        Returns interpolation of decoded latent space between 4 reconstructed images.
        '''
        latent = encoder(self.grid_images)
        corners = latent.permute(1, 0, 2, 3).view(self.size_z, 1, 2, 2)
        interpolated = F.interpolate(corners, size=6, mode='bilinear', align_corners=True)
        images = decoder(interpolated.view(self.size_z, 36, 1, 1).permute(1, 0, 2, 3))
        return make_grid(images, nrow=6, normalize=True)

    @torch.no_grad()
    def log_summary(self, step, writer, model):
        '''
        Log epoch summary to tensorboard writer.
        '''
        encoder = model.encode
        decoder = model.decode

        # SWD metric
        swd_metrics = self.swd(decoder)
        for level, value in enumerate(swd_metrics):
            writer.add_scalar(f'swd/swd_level{level}', value, global_step=step)

        # Reconstruction quality indicators
        images_compare, l2_distance = self.reconstruct(encoder, decoder)
        writer.add_image('reconstruct_images', images_compare, global_step=step)
        writer.add_scalar('reconstruct_l2', l2_distance, global_step=step)

        # Latent space quality indicators
        writer.add_image('generated', self.generate(decoder), global_step=step)
        writer.add_image('interpolate_noise', self.interpolate_noise(decoder), global_step=step)
        writer.add_image('interpolate_images', self.interpolate_images(encoder, decoder), global_step=step)


class ClassifierSummary():

    def __init__(self, dataset, **kwargs):
        '''
        Encoder/Decoder shapes:
        Decoder function: z (B, size_z, 1, 1) -> image (B, C, W, H)
        Encoder function: image (B, C, W, H) -> z (B, size_z, 1, 1)
        '''
        self.dataset = dataset

    @torch.no_grad()
    def log_summary(self, step, writer, model):
        writer.add_scalar(f'accuracy', self.accuracy(model), global_step=step)
    
    def accuracy(self, classifier):
        no_correct = 0
        loader = DataLoader(self.dataset, batch_size=1024)
        for images, labels in loader:
            output = classifier(images)
            predictions = output.data.argmax(dim=1)
            no_correct += predictions.eq(labels).sum().item()

        return no_correct / len(self.dataset)


def inference_mode(function, model, device='cpu'):
    '''
    Wraps function with model in inference mode. 
    Set model back to its initial state after the function call.
    '''
    def inner_function(*args, **kwargs):

        # Save initial settings
        model_training =  model.training
        model_device = model.device
        
        # Call function in eval mode
        model.to(device)
        model.eval()
        with torch.no_grad():
            output = function(*args, **kwargs)

        # Return model to initial settings
        model.to(model_device)
        if model_training:
            model.train()
        return output

    return inner_function