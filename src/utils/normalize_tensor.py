import torch


def normalize_tensor(tensor):
    max, min = torch.max(tensor), torch.min(tensor)

    numerator = tensor - min
    denominator = max - min

    return numerator / denominator