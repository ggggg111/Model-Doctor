import os
import argparse

import torch

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm

from utils import load_dataset
from utils import load_model


parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_path", type=str, default=os.path.join("d:", "Datasets", "Detection"),
    help="Directory path to the dataset"
)

parser.add_argument(
    "--dataset", type=str,
    choices=("mnist", "fashion_mnist", "cifar10", "cifar100", "svhn", "stl10"),
    default="fashion_mnist",
    help="Name of the dataset"
)

parser.add_argument(
    "--device", type=str, choices=("cuda", "cpu"), default="cuda",
    help="Either use cuda or cpu"
)

parser.add_argument(
    "--model_name", type=str,
    choices=(
        "alexnet", "vgg16", "resnet50", "wide_resnet50_2", "resnext50_32x4d", "densenet121", "efficientnet_b2",
        "googlenet", "mobilenet_v2", "inception_v3", "shufflenet_v2_x1_0", "squeezenet1_0", "mnasnet1_0"
    ),
    default="alexnet",
    help="Which model to use"
)

parser.add_argument(
    "--checkpoints_path", type=str, default="checkpoints",
    help="Path to where to save the trained model"
)

parser.add_argument(
    "--batch_size", type=int, default=64,
    help="Batch size used for training"
)

parser.add_argument(
    "--learning_rate", type=float, default=1e-2,
    help="Learning rate used for training"
)

parser.add_argument(
    "--epochs", type=int, default=100,
    help="Maximum epochs used for training"
)

args = parser.parse_args()


def main():
    data_path = args.data_path
    dataset = args.dataset
    device = torch.device(args.device)
    model_name = args.model_name
    checkpoints_path = os.path.join(args.checkpoints_path, dataset)
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    train_data = load_dataset(data_path, dataset, "train")
    num_classes = len(train_data.classes)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    model = load_model(model_name, num_classes, device)
    model.train()

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loss_epoch = 0

        for data, targets in tqdm(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

        print(f"Epoch: [{epoch + 1}/{epochs}], Loss: {loss_epoch / len(train_loader)}")

    torch.save(model.state_dict(), os.path.join(checkpoints_path, f"{model_name}.pt"))


if __name__ == "__main__":
    main()