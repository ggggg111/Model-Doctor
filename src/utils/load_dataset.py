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
                transform = T.Compose([T.Grayscale(3), T.Resize(256), T.ToTensor()])
            data = MNIST(root=path, train=True, download=False, transform=transform)
        elif split == "test":
            if transform is None:
                transform = T.Compose([T.Grayscale(3), T.Resize(256), T.ToTensor()])
            data = MNIST(root=path, train=False, download=False, transform=transform)

    elif name == "fashion_mnist":
        if split == "train":
            if transform is None:
                transform = T.Compose([T.Grayscale(3), T.Resize(256), T.ToTensor()])
            data = FashionMNIST(root=path, train=True, download=False, transform=transform)
        elif split == "test":
            if transform is None:
                transform = T.Compose([T.Grayscale(3), T.Resize(256), T.ToTensor()])
            data = FashionMNIST(root=path, train=False, download=False, transform=transform)

    elif name == "cifar10":
        if split == "train":
            if transform is None:
                transform = T.Compose([T.Grayscale(3), T.Resize(256), T.ToTensor()])
            data = CIFAR10(root=path, train=True, download=False, transform=transform)
        elif split == "test":
            if transform is None:
                transform = T.Compose([T.Grayscale(3), T.Resize(256), T.ToTensor()])
            data = CIFAR10(root=path, train=False, download=False, transform=transform)

    elif name == "cifar100":
        if split == "train":
            if transform is None:
                transform = T.Compose([T.Grayscale(3), T.Resize(256), T.ToTensor()])
            data = CIFAR100(root=path, train=True, download=False, transform=transform)
        elif split == "test":
            if transform is None:
                transform = T.Compose([T.Grayscale(3), T.Resize(256), T.ToTensor()])
            data = CIFAR100(root=path, train=False, download=False, transform=transform)

    elif name == "svhn":
        if split == "train":
            if transform is None:
                transform = T.Compose([T.Grayscale(3), T.Resize(256), T.ToTensor()])
            data = SVHN(root=path, split=split, download=False, transform=transform)
        elif split == "test":
            if transform is None:
                transform = T.Compose([T.Grayscale(3), T.Resize(256), T.ToTensor()])
            data = SVHN(root=path, split=split, download=False, transform=transform)

    elif name == "stl10":
        if split == "train":
            if transform is None:
                transform = T.Compose([T.Grayscale(3), T.Resize(256), T.ToTensor()])
            data = STL10(root=path, split=split, download=False, transform=transform)
        elif split == "test":
            if transform is None:
                transform = T.Compose([T.Grayscale(3), T.Resize(256), T.ToTensor()])
            data = STL10(root=path, split=split, download=False, transform=transform)

    return data