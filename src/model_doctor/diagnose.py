import os

import torch
import torch.nn as nn
import numpy as np

from utils import load_model_layers
from .gradient_hook import GradientHook
from .noise_hook import NoiseHook


class DiagnoseStage:
    def __init__(self, model, num_classes, delta, device):
        self.model = model
        self.num_classes = num_classes
        self.delta = delta
        self.device = device

        self.conv2d_layers = load_model_layers(self.model, nn.Conv2d)

        self.modules_gradient = [GradientHook(module, self.device) for module in self.conv2d_layers]
        self.modules_noise = [NoiseHook(module, self.delta, self.device) for module in self.conv2d_layers]

        self.gradients = [[[] for _ in range(self.num_classes)] for _ in range(len(self.conv2d_layers))]

    def compute_gradients(self, outputs, classes):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, classes)

        for module_index, module_gradient in enumerate(self.modules_gradient):
            gradients = torch.autograd.grad(loss, module_gradient.output, retain_graph=True)
            gradients = torch.squeeze(gradients[0])
            gradients = torch.abs(gradients)
            gradients = torch.sum(gradients, dim=(1, 2))

            self.gradients[module_index][classes.item()].append(gradients.detach().cpu().numpy())

    def save_gradients(self, gradients_path):
        for module_index, module_gradient_sum in enumerate(self.gradients):
            module_gradient_sum = np.asarray(module_gradient_sum)
            module_gradient_sum_mean = np.mean(module_gradient_sum, axis=1)

            np.save(os.path.join(gradients_path, f"layer_{module_index}"), module_gradient_sum_mean)

    def apply_noise(self):
        for module_noise in self.modules_noise:
            module_noise.apply_noise_hook()

    def remove_noise(self):
        for module_noise in self.modules_noise:
            module_noise.remove_noise_hook()