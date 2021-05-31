import numpy as np
import torch
import torch.nn.functional as F


def gaussian_kernel(channels, device="cpu"):
    kernel = torch.FloatTensor([
        [1, 4,  6,  4,  1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4,  6,  4,  1]]) / 256.0
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel.to(device)

def downsample(image, device="cpu"):
    channels = image.shape[1]
    kernel = gaussian_kernel(channels, device)
    image = F.pad(image, (2, 2, 2, 2), mode='replicate')
    return F.conv2d(image, kernel, groups=channels, stride=2)

def upsample(image, device="cpu"):
    channels = image.shape[1]
    kernel = gaussian_kernel(channels, device)
    image = F.interpolate(image, scale_factor=2) # Linear instead of zeroes
    image = F.pad(image, (2, 2, 2, 2), mode='replicate')
    return F.conv2d(image, kernel, groups=channels)

def laplacian_pyramid(image, levels, device="cpu"):
    '''
    Calculate Laplacian pyramid. Smallest level is without differing to enable reconstruction.
    '''
    current = image
    pyramid = []
    for level in range(levels-1):
        down = downsample(current, device)
        up = upsample(down, device)
        diff = current - up
        pyramid.append(diff)
        current = down
    pyramid.append(current) # last is without difference
    return pyramid

def normalize_channel(tensor):
    '''
    Normalize by channel mean and std. Tensor shape: (B, C, H, W)
    '''
    mean = tensor.mean(axis=(0, 2, 3), keepdim=True)
    std = tensor.std(axis=(0, 2, 3), keepdim=True)
    return (tensor - mean) / std

def get_all_patches(tensor, patch_size):
    '''
    Given image tensor and patch size, create tensor with all posible square patches.

    Input:  tensor of images in shape (B, C, H, W)
    Output: tensor of patches, in shape: (B, (H - patch_size) * (W - patch_size), C, H, W)
    '''
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    
    patches = tensor.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    patches = patches.reshape(batch_size, channels, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)
    return patches

def get_descriptors(tensor1, tensor2, n_patches, patch_size):
    '''
    Given two image tensors create descriptors from n matching patches with given patch_size.

    Input:  two tensors of shape: (B, C, H, W)
    Output: two tensors of shape: (B * n_patches, C * patch_size * patch_size)
    '''
    assert tensor1.shape == tensor2.shape
    channels = tensor1.shape[1]

    all_patches1 = get_all_patches(tensor1, patch_size)
    all_patches2 = get_all_patches(tensor2, patch_size)
    patches_size = all_patches1.shape[1]

    # Get random patches
    random_indexes = torch.randperm(patches_size)[:n_patches]
    d1 = all_patches1[:, random_indexes].view(-1, channels, patch_size, patch_size)
    d2 = all_patches2[:, random_indexes].view(-1, channels, patch_size, patch_size)
    
    # Normalize by channel mean/std and flatten
    d1 = normalize_channel(d1).view(-1, channels*patch_size*patch_size)
    d2 = normalize_channel(d2).view(-1, channels*patch_size*patch_size)
    return d1, d2

def sliced_wasserstein(A, B, n_projections, projection_size, device='cpu'):
    '''
    Calculates sliced wasserstein distance between two tensors.
    
    Input: two tensors of shape: (B, Values)
    Output: SWD distance
    '''
    assert A.ndim == 2 and A.shape == B.shape

    distances = []
    for i in range(n_projections):
        noise = torch.randn(A.shape[1], projection_size).to(device)
        noise = noise / torch.std(noise, dim=0, keepdim=True)

        # Projections
        projA = torch.matmul(A, noise)
        projB = torch.matmul(B, noise)
        projA, _ = torch.sort(projA, dim=0, descending=False)
        projB, _ = torch.sort(projB, dim=0, descending=False)
        mean_distance = torch.mean(torch.abs(projA - projB))
        distances.append(mean_distance)
    return torch.mean(torch.stack(distances))

@torch.no_grad()
def swd_metric(A, B, n_projections=4, projection_size=128, n_patches=128, patch_size=7, device='cuda'):
    '''
    Calculates SWD on random patches for each level of Laplacian pyramid.

    Input:  two tensors A, B of shape: (B, C, H, W)
    Output: 1D tensor, values are metrics for each pyramid level.
    '''
    levels = int(max(np.log2(A.shape[2]) - np.log2(16), 0)) + 1 # Smallest pyramid is 16x16
    pyramidA = laplacian_pyramid(A, levels, device='cpu')
    pyramidB = laplacian_pyramid(B, levels, device='cpu')

    distances = []
    for i in range(levels):
        descA, descB = get_descriptors(pyramidA[i], pyramidB[i], n_patches, patch_size)
        swd = sliced_wasserstein(descA.to(device), descB.to(device), n_projections, projection_size, device)
        distances.append(swd.to('cpu'))
    return torch.stack(distances)

@torch.no_grad()
def is_metric(P, eps=1e-10):
    '''
    Calculate metric based on Inception Score.
    KL Divergence between conditional and marginal distribtuions.

    Input:  tensor P of logits of shape: (B, No of Classes)
    Output: scalar
    '''    
    kld = P * (torch.log(P + eps) - torch.log(P.mean(axis=0) + eps))
    kld = torch.exp(kld.sum(axis=1).mean())
    return kld


def cross_entropy_loss(predictions, targets):
    '''
    Re-implementation of Pytorch CrossEntropyLoss to work with one-hot inputs.
    '''
    ce = torch.sum(-targets*F.log_softmax(predictions, dim=1), dim=1)
    return torch.mean(ce)