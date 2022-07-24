import os
import argparse

import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model_doctor import DiagnoseStage
from utils import load_model


parser = argparse.ArgumentParser()

parser.add_argument(
    "--high_confidence_samples_path", type=str, default="high_confidence_samples",
    help="Directory path to the high confidence images"
)

parser.add_argument(
    "--dataset", type=str,
    choices=("mnist", "fashion_mnist", "cifar10", "cifar100", "svhn", "stl10"),
    default="cifar10",
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
    default="resnet50",
    help="Which model to use"
)

parser.add_argument(
    "--checkpoints_path", type=str, default="checkpoints",
    help="Path to where the checkpoint is stored"
)

parser.add_argument(
    "--checkpoint_file", type=str, default="resnet50.pt",
    help="Name of the checkpoint file used, including the extension"
)

parser.add_argument(
    "--gradients_path", type=str, default="gradients",
    help="Path to where the gradients will be stored"
)

parser.add_argument(
    "--delta", type=int, default=0.1,
    help="Delta value for the noise"
)


args = parser.parse_args()


def main():
    high_confidence_samples_path = args.high_confidence_samples_path
    dataset = args.dataset
    device = torch.device(args.device)
    model_name = args.model_name
    high_confidence_samples_path = os.path.join(high_confidence_samples_path, dataset, model_name)
    checkpoints_path = os.path.join(args.checkpoints_path, dataset)
    checkpoint_file = args.checkpoint_file
    checkpoint_path = os.path.join(checkpoints_path, checkpoint_file)
    gradients_path = os.path.join(args.gradients_path, dataset, model_name)
    delta = args.delta

    if not os.path.exists(gradients_path):
        os.makedirs(gradients_path)

    transform = T.Compose([
        T.ToTensor()
    ])

    high_confidence_data = ImageFolder(
        root=high_confidence_samples_path,
        transform=transform,
    )

    num_classes = len(high_confidence_data.classes)

    high_confidence_loader = DataLoader(
        high_confidence_data,
        shuffle=False
    )

    model = load_model(model_name, num_classes, device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    diagnose_stage = DiagnoseStage(model, num_classes, delta, device)

    for data, targets in tqdm(high_confidence_loader):
        data = data.to(device)
        targets = targets.to(device)

        diagnose_stage.apply_noise()
        outputs = model(data)
        diagnose_stage.remove_noise()

        diagnose_stage.compute_gradients(outputs, targets)

    diagnose_stage.save_gradients(gradients_path)


if __name__ == "__main__":
    main()