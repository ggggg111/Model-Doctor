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

        self.module_gradient = GradientHook(self.conv2d_layers[-1], self.device)
        self.module_noise = NoiseHook(self.conv2d_layers[-1], self.delta, self.device)

        self.gradients = torch.from_numpy(load_model_gradients(self.gradients_path)[-1])

    def channel_loss(self, outputs, targets, threshold=1.0):
        loss_channel = 0

        criterion = nn.CrossEntropyLoss()
        loss_target = criterion(outputs, targets)

        gradients_target = torch.autograd.grad(loss_target, self.module_gradient.output, retain_graph=True)
        gradients_target = torch.abs(gradients_target[0])
        gradients_target = torch.sum(gradients_target, dim=(2, 3))

        outputs_sm = F.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs_sm, dim=1)

        loss_prediction = criterion(outputs, predictions)

        gradients_prediction = torch.autograd.grad(loss_prediction, self.module_gradient.output, retain_graph=True)
        gradients_prediction = torch.abs(gradients_prediction[0])
        gradients_prediction = torch.sum(gradients_prediction, dim=(2, 3))

        first_term_sum = 0

        for target, gradients_target_ind in zip(targets, gradients_target):
            for feature_map_ind_sum, feature_map_avg_sum in zip(gradients_target_ind, self.gradients[target]):
                multiplier = 1.0 if feature_map_avg_sum < threshold else 0.0
                first_term_sum += multiplier * feature_map_ind_sum

        second_term_sum = torch.sum(gradients_prediction)

        loss_channel += first_term_sum + second_term_sum

        return loss_channel / outputs.shape[0]

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
            sum = torch.sum(hadamard_product)

            loss_spatial += sum

        return loss_spatial / outputs.shape[0]

    def apply_noise(self):
        self.module_noise.apply_noise_hook()

    def remove_noise(self):
        self.module_noise.remove_noise_hook()