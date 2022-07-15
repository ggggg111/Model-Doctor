from torchvision.models import alexnet
from torchvision.models import vgg16
from torchvision.models import resnet50
from torchvision.models import wide_resnet50_2
from torchvision.models import resnext50_32x4d
from torchvision.models import densenet121
from torchvision.models import efficientnet_b2
from torchvision.models import googlenet
from torchvision.models import mobilenet_v2
from torchvision.models import inception_v3
from torchvision.models import shufflenet_v2_x1_0
from torchvision.models import squeezenet1_0
from torchvision.models import mnasnet1_0


def load_model(model_name, num_classes, device):
    model = None

    if model_name == "alexnet":
        model = alexnet(pretrained=False, progress=False, num_classes=num_classes).to(device)

    elif model_name == "vgg16":
        model = vgg16(pretrained=False, progress=False, num_classes=num_classes).to(device)

    elif model_name == "resnet50":
        model = resnet50(pretrained=False, progress=False, num_classes=num_classes).to(device)

    elif model_name == "wide_resnet50_2":
        model = wide_resnet50_2(pretrained=False, progress=False, num_classes=num_classes).to(device)

    elif model_name == "resnext50_32x4d":
        model = resnext50_32x4d(pretrained=False, progress=False, num_classes=num_classes).to(device)

    elif model_name == "densenet121":
        model = densenet121(pretrained=False, progress=False, num_classes=num_classes).to(device)

    elif model_name == "efficientnet_b2":
        model = efficientnet_b2(pretrained=False, progress=False, num_classes=num_classes).to(device)

    elif model_name == "googlenet":
        model = googlenet(pretrained=False, progress=False, num_classes=num_classes).to(device)

    elif model_name == "mobilenet_v2":
        model = mobilenet_v2(pretrained=False, progress=False, num_classes=num_classes).to(device)

    if model_name == "inception_v3":
        model = inception_v3(pretrained=False, progress=False, num_classes=num_classes).to(device)

    if model_name == "shufflenet_v2_x1_0":
        model = shufflenet_v2_x1_0(pretrained=False, progress=False, num_classes=num_classes).to(device)

    if model_name == "squeezenet1_0":
        model = squeezenet1_0(pretrained=False, progress=False, num_classes=num_classes).to(device)

    if model_name == "mnasnet1_0":
        model = mnasnet1_0(pretrained=False, progress=False, num_classes=num_classes).to(device)

    return model