import torchvision.transforms as T

from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import SVHN
from torchvision.datasets import STL10


def load_dataset(path, name, split, transform=None):
    data = None

    if name == "mnist":
        if split == "train":
            if transform is None:
                transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.Grayscale(3), T.Resize(128), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data = MNIST(root=path, train=True, download=False, transform=transform)
        elif split == "test":
            if transform is None:
                transform = T.Compose([T.Grayscale(3), T.Resize(128), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data = MNIST(root=path, train=False, download=False, transform=transform)

    elif name == "fashion_mnist":
        if split == "train":
            if transform is None:
                transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.Grayscale(3), T.Resize(128), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data = FashionMNIST(root=path, train=True, download=False, transform=transform)
        elif split == "test":
            if transform is None:
                transform = T.Compose([T.Grayscale(3), T.Resize(128), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data = FashionMNIST(root=path, train=False, download=False, transform=transform)

    elif name == "cifar10":
        if split == "train":
            if transform is None:
                transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.Resize(128), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data = CIFAR10(root=path, train=True, download=False, transform=transform)
        elif split == "test":
            if transform is None:
                transform = T.Compose([T.Resize(128), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data = CIFAR10(root=path, train=False, download=False, transform=transform)

    elif name == "cifar100":
        if split == "train":
            if transform is None:
                transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.Resize(128), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data = CIFAR100(root=path, train=True, download=False, transform=transform)
        elif split == "test":
            if transform is None:
                transform = T.Compose([T.Resize(128), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data = CIFAR100(root=path, train=False, download=False, transform=transform)

    elif name == "svhn":
        if split == "train":
            if transform is None:
                transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.Resize(128), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data = SVHN(root=path, split=split, download=False, transform=transform)
        elif split == "test":
            if transform is None:
                transform = T.Compose([T.Resize(128), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data = SVHN(root=path, split=split, download=False, transform=transform)

    elif name == "stl10":
        if split == "train":
            if transform is None:
                transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.Resize(128), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data = STL10(root=path, split=split, download=False, transform=transform)
        elif split == "test":
            if transform is None:
                transform = T.Compose([T.Resize(128), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data = STL10(root=path, split=split, download=False, transform=transform)

    return data