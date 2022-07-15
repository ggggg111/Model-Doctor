import argparse

from utils.annotation_to_mask import label_to_mask


parser = argparse.ArgumentParser()

parser.add_argument(
    "--low_confidence_annotations_path", type=str, default="low_confidence_annotations",
    help="Directory path to the low confidence annotations"
)

parser.add_argument(
    "--low_confidence_masks_path", type=str, default="low_confidence_masks",
    help="Directory path to the low confidence masks"
)

parser.add_argument(
    "--dataset", type=str,
    choices=("mnist", "fashion_mnist", "cifar10", "cifar100", "svhn", "stl10"),
    default="fashion_mnist",
    help="Name of the dataset"
)

args = parser.parse_args()


def main():
    low_confidence_annotations_path = args.low_confidence_annotations_path
    low_confidence_masks_path = args.low_confidence_masks_path
    dataset = args.dataset

    label_to_mask(low_confidence_annotations_path, low_confidence_masks_path, dataset)

if __name__ == "__main__":
    main()