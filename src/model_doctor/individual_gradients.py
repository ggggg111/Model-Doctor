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

    def compute_gradients(self, outputs, classes):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, classes)

        for module_index, module_gradient in enumerate(self.modules_gradient):
            gradients = torch.autograd.grad(loss, module_gradient.output, retain_graph=True)
            gradients = torch.squeeze(gradients[0])
            gradients = torch.abs(gradients)
            gradients = torch.sum(gradients, dim=(1, 2))

            self.gradients[module_index][classes.item()].append(gradients.detach().cpu().numpy())

    def compute_individual_gradients(self, single_output, single_class):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(single_output, single_class)

        last_module_gradient = self.modules_gradient[-1]

        gradients = torch.autograd.grad(loss, last_module_gradient.output, retain_graph=True)
        gradients = torch.squeeze(gradients[0])
        gradients = torch.abs(gradients)
        gradients = torch.sum(gradients, dim=0)

        gradients -= gradients.min()
        gradients /= gradients.max()

        return gradients