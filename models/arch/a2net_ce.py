import torch
import torch.nn as nn
import torch.nn.functional as F

from models.arch.base import BaseNet
from models.backbone.clip import CLIPVision
from models.backbone.resnet import ResNet50


class A2NetCE(BaseNet):
    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 num_attns: int = 4,
                 with_softplus: bool = False,
                 **kwargs):
        super().__init__(backbone, nbit, nclass, **kwargs)
        assert isinstance(backbone, (CLIPVision, ResNet50))

        if with_softplus:
            self.attn_conv = nn.Sequential(nn.Conv2d(self.backbone.features_size, num_attns, 1, 1, 0),
                                           nn.Softplus())
        else:
            self.attn_conv = nn.Conv2d(self.backbone.features_size, num_attns, 1, 1, 0)

        self.local_conv = nn.Conv2d(self.backbone.features_size, self.backbone.features_size, 1, 1, 0)
        self.global_conv = nn.Conv2d(self.backbone.features_size, self.backbone.features_size, 1, 1, 0)
        self.hash_fc = nn.Linear(self.backbone.features_size * (num_attns + 1), self.nbit, bias=False)
        self.ce_fc = nn.Linear(self.nbit, self.nclass)
        self.num_attns = num_attns

    def get_training_modules(self):
        return nn.ModuleDict({'attn_conv': self.attn_conv,
                              'local_conv': self.local_conv,
                              'global_conv': self.global_conv,
                              'hash_fc': self.hash_fc,
                              'ce_fc': self.ce_fc})

    def get_attn_maps(self, x):
        x = self.backbone.forward_feature_maps(x)  # (B, C, H, W)
        attn = self.attn_conv(x)  # (B, 4, H, W)
        return attn

    def forward(self, x, return_codes_and_attn_only=False):
        """

        :param x:
        :return:
            u = logits
            h = code logits
        """
        x = self.backbone.forward_feature_maps(x)  # (B, C, H, W)

        attn = self.attn_conv(x)  # (B, 4, H, W)
        attn_x = []
        for i in range(self.num_attns):
            attn_x.append(self.local_conv(attn[:, i:i + 1, :, :] * x))
        attn_x = torch.stack(attn_x, 1)  # (B, 4, C, H, W)
        all_x = torch.cat([attn_x, self.global_conv(x).unsqueeze(1)], dim=1)  # (B, 5, C, H, W)
        all_x = all_x.mean(dim=(3, 4))  # (B, 5, C)
        all_x = all_x.reshape(all_x.size(0), -1)

        # weight = (nbit, feat_size)
        codes = all_x @ self.hash_fc.weight.t()  # (B, feat_size) -> (B, nbit)
        codes_tanh = torch.tanh(codes)
        rec_all_x = codes_tanh @ self.hash_fc.weight  # (B, nbit) -> (B, feat_size)

        logits = self.ce_fc(codes_tanh)

        if return_codes_and_attn_only:
            return codes, attn

        return codes, codes_tanh, logits, all_x, rec_all_x


class TempCE(nn.Module):
    def __init__(self, center: torch.Tensor, nbit: int, temp: float = 10, nonlinear=True):
        super().__init__()
        self.center: torch.Tensor
        self.register_buffer('center', center)
        if nonlinear:
            self.tp = nn.Sequential(
                nn.Linear(center.size(1), center.size(1)),
                nn.ReLU(inplace=True),
                nn.Linear(center.size(1), nbit),
            )
        else:
            self.tp = nn.Linear(center.size(1), nbit)
        self.temp = temp

    def forward(self, x):
        w = self.tp(self.center)

        l2_x = F.normalize(x, dim=-1, p=2)
        l2_w = F.normalize(w, dim=-1, p=2)

        cont_logits = l2_x @ l2_w.t()
        return cont_logits * self.temp  # scale -1~1 to -10~10

    def extra_repr(self) -> str:
        return 'center={}'.format(
            self.center.shape
        )


class A2NetCEWithFixedPrompt(A2NetCE):
    def __init__(self, backbone: nn.Module, nbit: int, nclass: int, num_attns: int = 4, **kwargs):
        super().__init__(backbone, nbit, nclass, num_attns, **kwargs)

        self.ce_fc = TempCE(kwargs['fixed_center'], nbit, kwargs.get('temp', 10), kwargs.get('nonlinear', True))
