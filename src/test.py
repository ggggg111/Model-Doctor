import os
import argparse

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
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
    help="Path to where the checkpoints are stored"
)

parser.add_argument(
    "--checkpoint_file", type=str, default="alexnet.pt",
    help="Name of the checkpoint file used, including the extension"
)

parser.add_argument(
    "--batch_size", type=int, default=64,
    help="Batch size used for testing"
)

args = parser.parse_args()


def main():
    data_path = args.data_path
    dataset = args.dataset
    device = torch.device(args.device)
    model_name = args.model_name
    checkpoints_path = os.path.join(args.checkpoints_path, dataset)
    checkpoint_file = args.checkpoint_file
    checkpoint_path = os.path.join(checkpoints_path, checkpoint_file)
    batch_size = args.batch_size

    test_data = load_dataset(data_path, dataset, "test")
    num_classes = len(test_data.classes)

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    model = load_model(model_name, num_classes, device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in tqdm(test_loader):
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, dim=1)

            total += targets.shape[0]
            correct += int((predicted == targets).sum())

    print(f"Accuracy: {correct / total}")


if __name__ == "__main__":
    main()