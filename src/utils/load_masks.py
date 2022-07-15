import os

import cv2


def load_masks(masks_path, num_classes, masks_per_class):
    masks = [[None  for _ in range(masks_per_class)] for _ in range(num_classes)]

    for class_index, class_directory in enumerate(os.listdir(masks_path)):
        for mask_index, mask_file in enumerate(os.listdir(os.path.join(masks_path, class_directory))):
            mask_path = os.path.join(masks_path, class_directory, mask_file)

            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            masks[class_index][mask_index] = mask

    return masks