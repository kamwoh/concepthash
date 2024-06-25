from models.backbone.alexnet import AlexNet
from models.backbone.identity import Identity
from models.backbone.resnet import ResNet50, ResNet18
from models.backbone.swinvit import SwinViTBase, SwinViTSmall, SwinViTTiny
from models.backbone.vgg16 import VGG16
from models.backbone.vit import ViTBase, ViTSmall, ViTTiny


def get_backbone(name, **kwargs):
    """

    :param name:
        ['alexnet', 'vgg16']
    :param kwargs:
        pretrained=True
    :return:
    """
    if name == 'alexnet':
        return AlexNet(**kwargs)
    elif name == 'vgg16':
        return VGG16(**kwargs)
    elif name == 'identity':
        return Identity(**kwargs)
    elif name == 'resnet18':
        return ResNet18(**kwargs)
    elif name == 'resnet50':
        return ResNet50(**kwargs)
    elif name == 'vit-s16':  # small
        return ViTSmall(**kwargs)
    elif name == 'vit-b16':  # base
        return ViTBase(**kwargs)
    elif name == 'swinvit-s16':  # small
        return SwinViTSmall(**kwargs)
    elif name == 'swinvit-t16':  # tiny
        return SwinViTTiny(**kwargs)
    elif name == 'swinvit-b16':  # base
        return SwinViTBase(**kwargs)
    else:
        raise NotImplementedError(f'no implementation for {name}')
