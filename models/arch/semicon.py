import math

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchvision.models.resnet import (conv1x1, conv3x3)

from models.arch.base import BaseNet

# see https://github.com/aassxun/SEMICON/blob/main/models/SEMICON.py

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

"""
Two stage
"""


class ChannelTransformer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

        self.qkv = nn.Conv2d(dim, dim * 3, 1, groups=num_heads)
        self.qkv2 = nn.Conv2d(dim, dim * 3, 1, groups=head_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W).transpose(0, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, nh, hdim, h*w)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (b, nh, hdim, hdim)
        attn = torch.sign(attn) * torch.sqrt(torch.abs(attn) + 1e-5)
        attn = attn.softmax(dim=-1)

        # attn @ v = (b, nh, hdim, hdim) @ (b, nh, hdim, h*w) = (b, nh, hdim, h*w) -> (b, nh*hdim, h, w)
        x = ((attn @ v).reshape(B, C, H, W) + x)
        x = x.reshape(B, self.num_heads, self.head_dim, H, W).transpose(1, 2).reshape(B, C, H, W)

        y = self.norm(x)
        x = self.relu(y)
        qkv2 = self.qkv2(x).reshape(B, 3, self.head_dim, self.num_heads, H * W).transpose(0, 1)
        q, k, v = qkv2[0], qkv2[1], qkv2[2]

        attn = (q @ k.transpose(-2, -1)) * (self.num_heads ** -0.5)
        attn = torch.sign(attn) * torch.sqrt(torch.abs(attn) + 1e-5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, self.head_dim, self.num_heads, H, W).transpose(1, 2).reshape(B, C, H, W) + y
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_Backbone(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Backbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def SEMICON_backbone(pretrained=True, progress=True, **kwargs):
    model = ResNet_Backbone(Bottleneck, [3, 4, 6], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=progress)
        for name in list(state_dict.keys()):
            if 'fc' in name or 'layer4' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict)
    return model


class ResNet_Refine(nn.Module):

    def __init__(self, block, layer, is_local=True, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Refine, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 1024
        self.dilation = 1

        self.is_local = is_local
        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(block, 512, layer, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
            layers.append(ChannelTransformer(planes * block.expansion, max(planes * block.expansion // 64, 16)))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer4(x)

        pool_x = self.avgpool(x)
        pool_x = torch.flatten(pool_x, 1)
        if self.is_local:
            return x, pool_x
        else:
            return pool_x

    def forward(self, x):
        return self._forward_impl(x)


def SEMICON_refine(is_local=True, pretrained=True, progress=True, **kwargs):
    model = ResNet_Refine(Bottleneck, 3, is_local, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=progress)
        for name in list(state_dict.keys()):
            if not 'layer4' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict, strict=False)
    return model


class SEM(nn.Module):

    def __init__(self, block, layer, att_size=4, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(SEM, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 1024
        self.dilation = 1
        self.att_size = att_size
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer4 = self._make_layer(block, 512, layer, stride=1)

        self.attn_convs = nn.ModuleList([nn.Sequential(
            conv1x1(self.inplanes, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        ) for _ in range(self.att_size)])
        #
        # self.feature1 = nn.Sequential(
        #     conv1x1(self.inplanes, 1),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(inplace=True),
        # )
        # self.feature2 = nn.Sequential(
        #     conv1x1(self.inplanes, 1),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(inplace=True)
        # )
        # self.feature3 = nn.Sequential(
        #     conv1x1(self.inplanes, 1),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(inplace=True)
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        att_expansion = 0.25
        layers = []
        layers.append(block(self.inplanes, int(self.inplanes * att_expansion), stride,
                            downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                conv1x1(self.inplanes, int(self.inplanes * att_expansion)),
                nn.BatchNorm2d(int(self.inplanes * att_expansion))
            ))
            self.inplanes = int(self.inplanes * att_expansion)
            layers.append(block(self.inplanes, int(self.inplanes * att_expansion), groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _mask(self, feature, x):
        with torch.no_grad():
            cam1 = feature.mean(1)
            attn = torch.softmax(cam1.view(x.shape[0], x.shape[2] * x.shape[3]), dim=1)  # B,H,W
            std, mean = torch.std_mean(attn)
            attn = (attn - mean) / (std ** 0.3) + 1  # 0.15
            attn = (attn.view((x.shape[0], 1, x.shape[2], x.shape[3]))).clamp(0, 2)
        return attn

    def _forward_impl(self, x):
        x = self.layer4(x)  # bs*64*14*14

        ys = []
        for i in range(self.att_size):
            y = self.attn_convs[i](x)
            if i != self.att_size - 1:
                att = 2 - self._mask(y, x)
                x = x.mul(att.repeat(1, self.inplanes, 1, 1))
            ys.append(y)

        ys = torch.cat(ys, dim=1)
        return ys

    def forward(self, x):
        return self._forward_impl(x)


def SEMICON_attention(att_size=3, pretrained=False, progress=True, **kwargs):
    model = SEM(Bottleneck, 3, att_size=att_size, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=progress)
        for name in list(state_dict.keys()):
            if 'fc' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict)
    return model


"""
Visual
"""


class SEMICON(nn.Module):
    def __init__(self,
                 nbit: int,
                 attn_size: int = 3,
                 feat_size: int = 2048,
                 **kwargs):
        super(SEMICON, self).__init__()

        pretrained = True
        code_length = nbit
        self.nbit = nbit
        self.feat_size = feat_size

        self.attn_size = attn_size

        self.backbone = SEMICON_backbone(pretrained=pretrained)
        self.refine_global = SEMICON_refine(is_local=False, pretrained=pretrained)
        self.refine_local = SEMICON_refine(pretrained=pretrained)
        self.attention = SEMICON_attention(att_size=attn_size)

        self.hash_layer_active = nn.Sequential(
            nn.Tanh(),
        )
        self.code_length = code_length

        remaining = code_length - ((code_length // 2) + 3 * (code_length // 6))
        self.W_G = nn.Parameter(torch.Tensor(code_length // 2 + remaining, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))

        # local
        self.W_L1 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_L1, a=math.sqrt(5))
        self.W_L2 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_L2, a=math.sqrt(5))
        self.W_L3 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_L3, a=math.sqrt(5))

        self.W = nn.ParameterDict({'W_G': self.W_G, 'W_L1': self.W_L1, 'W_L2': self.W_L2, 'W_L3': self.W_L3})

        self.bernoulli = torch.distributions.Bernoulli(0.5)

    def count_parameters(self, mode='trainable'):
        if mode == 'trainable':
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        elif mode == 'non-trainable':
            return sum(p.numel() for p in self.parameters() if not p.requires_grad)
        else:  # all
            return sum(p.numel() for p in self.parameters())

    def get_backbone(self):
        return self.backbone

    def get_training_modules(self):
        return nn.ModuleDict({
            'refine_global': self.refine_global,
            'refine_local': self.refine_local,
            'attention': self.attention,
            'W': self.W
        })

    def forward(self, x):
        layer3 = self.backbone(x)  # .detach()
        batch_size, channels, h, w = layer3.shape
        layer4 = self.refine_global(layer3)
        att_map = self.attention(layer3)  # batchsize * att-size * 14 * 14
        att_size = att_map.shape[1]
        att_map_rep = att_map.unsqueeze(axis=2).repeat(1, 1, channels, 1, 1)
        out_rep = layer3.unsqueeze(axis=1).repeat(1, att_size, 1, 1, 1)
        out_local = att_map_rep.mul(out_rep)
        out_local1 = out_local[:, :att_size // 3, :, :].reshape(batch_size * att_size // 3, channels, h, w)
        out_local2 = out_local[:, att_size // 3:att_size * 2 // 3, :, :].reshape(batch_size * att_size // 3, channels,
                                                                                 h, w)
        out_local3 = out_local[:, att_size * 2 // 3:att_size * 3 // 3, :, :].reshape(batch_size * att_size // 3,
                                                                                     channels, h, w)
        local_f1, avg_local_f1 = self.refine_local(out_local1)
        local_f2, avg_local_f2 = self.refine_local(out_local2)
        local_f3, avg_local_f3 = self.refine_local(out_local3)

        deep_S_G = F.linear(layer4, self.W_G)

        deep_S_1 = F.linear(avg_local_f1, self.W_L1)
        deep_S_2 = F.linear(avg_local_f2, self.W_L2)
        deep_S_3 = F.linear(avg_local_f3, self.W_L3)
        deep_S = torch.cat([deep_S_G, deep_S_1, deep_S_2, deep_S_3], dim=1)

        ret = self.hash_layer_active(deep_S)
        return ret, local_f1


class SEMICONWithAdapter(BaseNet):
    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 nattns: int = 4,
                 **kwargs):
        super().__init__(backbone, nbit, nclass, **kwargs)

        hidden_size = self.backbone.features_size
        self.backbone = self.backbone.model

        self.sem_attns = nn.ModuleList()
        self.icons = nn.ModuleList()
        self.hash_fcs = nn.ModuleList()
        for _ in range(nattns):
            attn = nn.Sequential(
                nn.Conv2d(hidden_size, 1, 1, 1, 0, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True)
            )
            self.sem_attns.append(attn)

            icon = ChannelTransformer(hidden_size, 16)
            self.icons.append(nn.Sequential(icon, nn.AdaptiveAvgPool2d(1), nn.Flatten()))

            self.hash_fcs.append(nn.Sequential(
                nn.Linear(hidden_size, self.nbit // (2 * nattns)),
                nn.Tanh()
            ))

        # for global
        icon = ChannelTransformer(hidden_size, 16)
        self.icons.append(nn.Sequential(icon, nn.AdaptiveAvgPool2d(1), nn.Flatten()))

        self.hash_fcs.append(nn.Sequential(
            nn.Linear(hidden_size, self.nbit // 2),
            nn.Tanh()
        ))

    def _mask(self, feature, x):
        with torch.no_grad():
            cam1 = feature.mean(1)
            attn = torch.softmax(cam1.view(x.shape[0], x.shape[2] * x.shape[3]), dim=1)  # B,H,W
            std, mean = torch.std_mean(attn)
            attn = (attn - mean) / (std ** 0.3) + 1  # 0.15
            attn = (attn.view((x.shape[0], 1, x.shape[2], x.shape[3]))).clamp(0, 2)
        return attn

    def forward_sem(self, x):
        attn = torch.ones_like(x)

        outs = []
        for i in range(len(self.sem_attns)):
            x = x * attn
            y = self.sem_attns[i](x)
            if i != len(self.sem_attns) - 1:
                attn = 2 - self._mask(y, x)
            outs.append(y)

        return torch.cat(outs, dim=1)

    def get_backbone(self):
        return nn.Identity()

    def get_training_modules(self):
        return nn.ModuleDict({  # 'trainable_params': self.trainable_params,
            'hash_fcs': self.hash_fcs,
            'sem_attns': self.sem_attns,
            'icons': self.icons
        })

    def forward(self, x):
        x = self.backbone(x).last_hidden_state[:, 1:, :]
        ntokens = int(x.size(1) ** 0.5)
        x = x.transpose(1, 2).reshape(x.size(0), -1, ntokens, ntokens)

        attn_map = self.forward_sem(x)

        codes = []
        for i in range(attn_map.size(1)):
            attn = attn_map[:, i:i + 1, :, :]
            attn_x = x * attn
            local_feat = self.icons[i](attn_x)
            local_hash = self.hash_fcs[i](local_feat)
            codes.append(local_hash)

        global_feat = self.icons[-1](x)
        global_hash = self.hash_fcs[-1](global_feat)
        codes.append(global_hash)
        codes = torch.cat(codes, dim=1)

        return codes, None


if __name__ == '__main__':
    model = SEMICON(attn_size=1,
                    nbit=64)
    print(model)
    print(model(torch.randn(1, 3, 224, 224)))
