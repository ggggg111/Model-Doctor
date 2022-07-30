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

    def channel_loss(self, outputs, threshold):
        loss_channel = 0

        criterion = nn.CrossEntropyLoss()
        outputs_sm = F.softmax(outputs, dim=1)

        _, best_two_predictions = torch.topk(outputs_sm, k=2, dim=1)
        best_predictions = torch.index_select(best_two_predictions, dim=1, index=torch.tensor([0]).to(self.device)).to(self.device)
        second_best_predictions = torch.index_select(best_two_predictions, dim=1, index=torch.tensor([1]).to(self.device)).to(self.device)

        best_predictions = torch.squeeze(best_predictions, 1)
        second_best_predictions = torch.squeeze(second_best_predictions, 1)

        loss_best_predictions = criterion(outputs, best_predictions)

        gradients_best_predictions = torch.autograd.grad(loss_best_predictions, self.module_gradient.output, retain_graph=True)
        gradients_best_predictions = torch.abs(gradients_best_predictions[0])
        gradients_best_predictions = torch.sum(gradients_best_predictions, dim=(2, 3))

        loss_second_best_predictions = criterion(outputs, second_best_predictions)

        gradients_second_best_predictions = torch.autograd.grad(loss_second_best_predictions, self.module_gradient.output, retain_graph=True)
        gradients_second_best_predictions = torch.abs(gradients_second_best_predictions[0])

        first_term_sum = 0

        for gradients_best_prediction, best_prediction in zip(gradients_best_predictions, best_predictions):
            for feature_map_pred_sum, feature_map_avg_sum in zip(gradients_best_prediction, self.gradients[best_prediction]):
                multiplier = 1.0 if feature_map_avg_sum < threshold else 0.0
                first_term_sum += multiplier * feature_map_pred_sum

        second_term_sum = torch.sum(gradients_second_best_predictions)

        loss_channel += first_term_sum + second_term_sum

        return loss_channel / outputs.shape[0]

    def spatial_loss(self, outputs, masks):
        loss_spatial = 0

        criterion = nn.CrossEntropyLoss()
        outputs_sm = F.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs_sm, dim=1)

        loss = criterion(outputs, predictions)

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