import os
import argparse

import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm

from model_doctor import TreatingStage
from utils import load_dataset
from utils import load_model
from utils import load_masks


parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_path", type=str, default=os.path.join("d:", "Datasets", "Detection", "CIFAR10"),
    help="Directory path to the dataset"
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
        "mobilenet_v2", "inception_v3", "shufflenet_v2_x1_0", "squeezenet1_0", "mnasnet1_0"
    ),
    default="resnet50",
    help="Which model to use"
)

parser.add_argument(
    "--checkpoints_path", type=str, default="checkpoints",
    help="Path to where the checkpoints are stored"
)

parser.add_argument(
    "--checkpoint_file", type=str, default="resnet50.pt",
    help="Name of the checkpoint file used, including the extension"
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
    "--epochs", type=int, default=50,
    help="Maximum epochs used for training"
)

parser.add_argument(
    "--gradients_path", type=str, default="gradients",
    help="Path to where the gradients are stored"
)

parser.add_argument(
    "--masks_path", type=str, default="low_confidence_masks",
    help="Path to where the masks are stored"
)

parser.add_argument(
    "--delta", type=int, default=0.1,
    help="Delta value for the noise"
)

parser.add_argument(
    "--threshold", type=float, default=1.0,
    help="Threshold value of channel loss"
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
    learning_rate = args.learning_rate
    epochs = args.epochs
    gradients_path = os.path.join(args.gradients_path, dataset, model_name)
    delta = args.delta
    threshold = args.threshold
    masks_path = os.path.join(args.masks_path, dataset, model_name)

    train_data = load_dataset(data_path, dataset, "train")
    num_classes = len(train_data.classes)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    mask_transform = T.Compose([
        T.ToTensor(),
        T.Resize(128)
    ])

    masks_per_class = len([image_path for image_path in os.listdir(os.path.join(masks_path, "class_0"))])
    masks = load_masks(masks_path, num_classes, masks_per_class)
    mask_counters = [0 for _ in range(num_classes)]

    model = load_model(model_name, num_classes, device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.train()

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)

    treating_stage = TreatingStage(model, gradients_path, delta, device)

    for epoch in range(epochs):
        loss_all_epoch = 0
        loss_original_epoch = 0
        loss_channel_epoch = 0
        loss_spatial_epoch = 0

        for data, targets in tqdm(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            mask_tensor_batch = []

            for target in targets:
                mask_tensor_batch.append(mask_transform(masks[target][mask_counters[target]]).to(device))
                mask_counters[target] = mask_counters[target] % (masks_per_class - 1) + 1

            treating_stage.apply_noise()
            outputs = model(data)
            treating_stage.remove_noise()

            loss_original = criterion(outputs, targets)
            loss_channel = treating_stage.channel_loss(outputs, threshold)
            loss_spatial = treating_stage.spatial_loss(outputs, mask_tensor_batch)

            loss_all = loss_original + loss_channel + loss_spatial

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            loss_all_epoch += loss_all.item()
            loss_original_epoch += loss_original.item()
            loss_channel_epoch += loss_channel.item()
            loss_spatial_epoch += loss_spatial.item()

        print(f"Epoch: [{epoch + 1}/{epochs}], Loss_all: {loss_all_epoch / len(train_loader)}, Loss_orig: {loss_original_epoch / len(train_loader)}")
        print(f"Loss_ch: {loss_channel_epoch / len(train_loader)}, Loss_sp: {loss_spatial_epoch / len(train_loader)}")

    torch.save(model.state_dict(), os.path.join(checkpoints_path, f"{model_name}_md.pt"))


if __name__ == "__main__":
    main()