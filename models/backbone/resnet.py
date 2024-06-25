import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, resnet101
from torchvision.models.resnet import BasicBlock, Bottleneck

from models.backbone.base import BaseNet


class ResNet50(BaseNet):
    def __init__(self, pretrained=True, **kwargs):
        super(ResNet50, self).__init__()

        model = resnet50(pretrained=pretrained)

        self.model = model
        self.features_size = model.fc.in_features

        del self.model.fc

    def set_stage4_stride1(self):
        for m in self.model.layer4.modules():
            if isinstance(m, BasicBlock):
                if isinstance(m.downsample, nn.Sequential):
                    m.downsample[0].stride = 1
                m.conv1.stride = 1
            elif isinstance(m, Bottleneck):
                if isinstance(m.downsample, nn.Sequential):
                    m.downsample[0].stride = 1
                m.conv2.stride = 1

    def forward_feature_maps(self, x):
        return self.forward(x, get_feat_map=True)

    def forward(self, x, get_feat_map=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        if get_feat_map:
            return x

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNet101(ResNet50):
    def __init__(self, pretrained=True, **kwargs):
        super(ResNet101, self).__init__()

        model = resnet101(pretrained=pretrained)

        self.model = model
        self.features_size = model.fc.in_features

        del self.model.fc


class ResNet18(BaseNet):
    def __init__(self, pretrained=True, **kwargs):
        super(ResNet18, self).__init__()

        model = resnet18(pretrained=pretrained)
        self.model = model

        self.features_size = model.fc.in_features

        del self.model.fc

    def train(self, mode: bool = True):
        super().train(mode)

        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

        return self

    def forward(self, x, get_feat_map=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        if get_feat_map:
            return x

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == '__main__':
    net = ResNet18(False)
    params = list(net.named_parameters())
    # print(params)
    for name, p in params:
        if p.requires_grad:
            print(name, p.requires_grad, p.size())
