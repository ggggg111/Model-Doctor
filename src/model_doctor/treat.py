import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_model_layers
from utils import load_model_gradients
from .gradient_hook import GradientHook
from .noise_hook import NoiseHook


class TreatingStage:
    def __init__(self, model, gradients_path, delta, device):
        self.model = model
        self.gradients_path = gradients_path
        self.delta = delta
        self.device = device

        self.conv2d_layers = load_model_layers(self.model, nn.Conv2d)

        self.modules_gradient = [GradientHook(module, self.device) for module in self.conv2d_layers]
        self.modules_noise = [NoiseHook(module, self.delta, self.device) for module in self.conv2d_layers]
        self.module_gradient = self.modules_gradient[-1]

        self.gradients = torch.from_numpy(load_model_gradients(self.gradients_path)[-1])

    def channel_loss(self, outputs, classes, threshold=1.0):
        loss_channel = 0

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, classes)

        gradients = torch.autograd.grad(loss, self.module_gradient.output, retain_graph=True)
        gradients = torch.abs(gradients[0])
        gradients = torch.sum(gradients, dim=(2, 3))

        outputs = F.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, dim=1)

        first_term_batch = 0

        loss_channel = torch.tensor([0]).to(self.device)

        #TODO

        return loss_channel

    def spatial_loss(self, outputs, classes, masks):


        return torch.tensor([0]).to(self.device)

    def apply_noise(self):
        for module_noise in self.modules_noise:
            module_noise.apply_noise_hook()

    def remove_noise(self):
        for module_noise in self.modules_noise:
            module_noise.remove_noise_hook()