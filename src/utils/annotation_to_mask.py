import os
import json

import cv2
import numpy as np

from labelme.utils import shape_to_mask


def label_to_mask(annotations_root, masks_root, dataset):
    annotations_path = os.path.join(annotations_root, dataset)

    for path, _, files in os.walk(annotations_path):
        save_file_path = path.replace(annotations_root, masks_root)
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)

        for file in files:
            file_path = os.path.join(path, file)

            with open(file_path, "r",encoding="utf-8") as f:
                img_json = json.load(f)

                mask = shape_to_mask(
                    img_shape=(img_json['imageHeight'], img_json['imageWidth']),
                    points=img_json['shapes'][0]['points'],
                    shape_type=None,
                    line_width=1,
                    point_size=1
                )

                mask = np.invert(mask)
                mask = mask.astype(np.int) * 255

                save_file_path = file_path.replace(annotations_root, masks_root)
                pre, _ = os.path.splitext(save_file_path)
                save_file_path = pre + ".png"

                cv2.imwrite(save_file_path, mask)