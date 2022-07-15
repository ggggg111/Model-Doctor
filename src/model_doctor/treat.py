import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from torchvision.transforms.functional import rgb_to_grayscale

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

    def channel_loss(self, outputs, targets, threshold=1.0):
        loss_channel = 0

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)

        gradients = torch.autograd.grad(loss, self.module_gradient.output, retain_graph=True)
        gradients = torch.abs(gradients[0])
        gradients = torch.sum(gradients, dim=(2, 3))

        outputs = F.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, dim=1)

        first_term_batch = 0

        loss_channel = torch.tensor([0]).to(self.device)

        #TODO

        return loss_channel

    def spatial_loss(self, outputs, targets, masks):
        loss_spatial = 0

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)

        gradients = torch.autograd.grad(loss, self.module_gradient.output, retain_graph=True)
        gradients = torch.abs(gradients[0])

        for gradient, mask in zip(gradients, masks):
            mask = rgb_to_grayscale(mask)
            mask = T.Resize((gradients.shape[2], gradients.shape[3]))(mask)
            mask = torch.squeeze(mask, 0)

            hadamard_product = torch.mul(mask, gradient)
            fmap_sum = torch.sum(hadamard_product, dim=(1, 2))
            k_sum = torch.sum(fmap_sum)

            loss_spatial += k_sum

        return loss_spatial / outputs.shape[0]

    def apply_noise(self):
        for module_noise in self.modules_noise:
            module_noise.apply_noise_hook()

    def remove_noise(self):
        for module_noise in self.modules_noise:
            module_noise.remove_noise_hook()