import os

import numpy as np


def load_model_gradients(path):
    return [np.load(os.path.join(path, file), allow_pickle=True) for file in os.listdir(path) if ".npy" in file]