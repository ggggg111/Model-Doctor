class GradientHook:
    def __init__(self, module, device):
        self.module = module
        self.device = device

        self.input = None
        self.output = None

        self.hook = self.module.register_forward_hook(self.__grad_hook)

    def remove_grad_hook(self):
        self.hook.remove()

    def __grad_hook(self, module, input, output):
        self.input = input
        self.output = output