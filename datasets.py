from torchvision import transforms, datasets

default_path = {
    'imagenet64': 'D:\datasets\imagenet-64\train_64x64',
    'celeba': 'D:\datasets\celeba\img_align_celeba',
    'mnist': 'D:\datasets',
    'cifar10': 'D:\datasets',
    'svhn': 'D:\datasets\SVHN',
    }

def get_dataset(dataset, path=None, train=True):
    if path is None:
        path = default_path[dataset]

    if dataset == 'imagenet64':
        t = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ]
        return datasets.ImageFolder(path, transform=transforms.Compose(t))

    elif dataset == 'celeba':
        t = [
            transforms.CenterCrop(160),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), #mean=[0.5123, 0.4166, 0.3665], std=[0.2976, 0.2702, 0.2661]
            ]
        
        return datasets.ImageFolder(path, transform=transforms.Compose(t))

    elif dataset == 'mnist':
        t = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]), # mean=[0.1307], std=[0.3081]
            ]
        return datasets.MNIST(path, train=train, transform=transforms.Compose(t))

    elif dataset == 'cifar10':
        t = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ]
        return datasets.CIFAR10(path, train=train, transform=transforms.Compose(t))

    elif dataset == 'svhn':
        t = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        if train:
            return datasets.SVHN(path, split='train', transform=transforms.Compose(t))
        else:
            return datasets.SVHN(path, split='test', transform=transforms.Compose(t))

    else:
        raise ValueError('Invalid dataset value')   