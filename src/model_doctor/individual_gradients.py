import torch
import torch.nn as nn

from utils import load_model_layers
from .gradient_hook import GradientHook


class IndividualGradients:
    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.conv2d_layers = load_model_layers(self.model, nn.Conv2d)

        self.modules_gradient = [GradientHook(module, self.device) for module in self.conv2d_layers]

    def compute_individual_gradients(self, single_output, single_class):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(single_output, single_class)

        gradients_layers = []

        for module_gradient in self.modules_gradient:
            gradients = torch.autograd.grad(loss, module_gradient.output, retain_graph=True)
            gradients = torch.squeeze(gradients[0])
            gradients = torch.abs(gradients)
            gradients = torch.sum(gradients, dim=0)

            gradients -= gradients.min()
            gradients /= gradients.max()

            gradients_layers.append(gradients)

        return gradients_layers