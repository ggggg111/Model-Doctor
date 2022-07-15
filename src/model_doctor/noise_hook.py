from torch.distributions.uniform import Uniform


class NoiseHook:
    def __init__(self, module, delta, device):
        self.module = module
        self.delta = delta
        self.device = device
        self.hook = None

    def apply_noise_hook(self):
        self.hook = self.module.register_forward_hook(self.__noise_hook)

    def remove_noise_hook(self):
        self.hook.remove()

    def __noise_hook(self, module, input, output):
        output += self.__generate_noise(output.shape)

    def __generate_noise(self, shape):
        return Uniform(-self.delta, self.delta).sample(shape).to(self.device)