import os
import argparse

import torch
import cv2
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns

from model_doctor import IndividualGradients
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
    "--image_path", type=str, required=True,
    help="Image path to view the statistical correlation between the category and convolution kernels"
)

parser.add_argument(
    "--image_class", type=int, required=True,
    help="Class of the image"
)

parser.add_argument(
    "--layer_index", type=int, required=True,
    help="Which 2D convolutional layer to view"
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
    image_path = args.image_path
    image_class = args.image_class
    layer_index = args.layer_index

    test_data = load_dataset(data_path, dataset, "test")
    num_classes = len(test_data.classes)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = T.Compose([
        T.ToTensor()
    ])

    image_tensor = transform(image)

    model = load_model(model_name, num_classes, device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    individual_gradients = IndividualGradients(model, device)

    class_tensor = torch.tensor(image_class).to(device).unsqueeze(0)

    image_tensor = image_tensor.to(device)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output = model(image_tensor)

    gradients_layers = individual_gradients.compute_individual_gradients(output, class_tensor)

    sns.heatmap(gradients_layers[layer_index].cpu().detach().numpy(), linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    main()