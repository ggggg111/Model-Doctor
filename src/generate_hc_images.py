import os
import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.utils import save_image
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
    "--high_confidence_path", type=str, default="high_confidence_samples",
    help="Directory to save the high confidence images"
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

parser.add_argument(
    "--num_images", type=int, default=100,
    help="Number of images to save for each class"
)

parser.add_argument(
    "--min_confidence", type=float, default=0.9,
    help="Minimum confidence required to save a sample (inclusive)"
)

args = parser.parse_args()


def main():
    data_path = args.data_path
    dataset = args.dataset
    high_confidence_path = os.path.join(args.high_confidence_path, dataset)
    device = torch.device(args.device)
    model_name = args.model_name
    checkpoints_path = os.path.join(args.checkpoints_path, dataset)
    checkpoint_file = args.checkpoint_file
    checkpoint_path = os.path.join(checkpoints_path, checkpoint_file)
    batch_size = args.batch_size
    num_images = args.num_images
    min_confidence = args.min_confidence

    if not os.path.exists(high_confidence_path):
        os.makedirs(high_confidence_path)

    transform = T.Compose([
        T.Grayscale(3),
        T.Resize(256),
        T.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.5, 0.75)),
        T.ToTensor()
    ])

    test_data = load_dataset(data_path, dataset, "test", transform)
    num_classes = len(test_data.classes)

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    model = load_model(model_name, num_classes, device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    class_path_dir = os.path.join(high_confidence_path, f"class_")
    for target in test_data.class_to_idx.values():
        class_path = class_path_dir + f"{target}"
        if not os.path.exists(class_path):
            os.makedirs(class_path)

    image_class_counters = np.zeros(num_classes, dtype=int)

    with torch.no_grad():
        for data, targets in tqdm(test_loader):
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            outputs = F.softmax(outputs, dim=1)
            confidences, predicted = torch.max(outputs, dim=1)

            for index, (confidence, prediction, target) in enumerate(zip(confidences, predicted, targets)):
                confidence = confidence.item()
                prediction = prediction.item()

                if confidence > min_confidence and image_class_counters[target] < num_images:
                    image_path = class_path_dir + f"{target}"
                    save_image(data[index], os.path.join(image_path, f"{image_class_counters[target]}" + ".png"))
                    image_class_counters[target] += 1


if __name__ == "__main__":
    main()