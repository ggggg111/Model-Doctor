def load_model_layers(model, type):
    return [module for module in model.modules() if isinstance(module, type)]