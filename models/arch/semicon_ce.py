import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.arch.base import BaseNet
from models.arch.semicon import SEMICON_refine, SEMICON_backbone, SEMICON_attention, ChannelTransformer
from models.layers.cossim import CosSim


class SEMICONCEWithAdapter(BaseNet):
    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 nattns: int = 4,
                 **kwargs):
        super().__init__(backbone, nbit, nclass, **kwargs)

        self.nattns = nattns
        # self.trainable_params = nn.ParameterDict()
        hidden_size = self.backbone.features_size

        self.downscale = self.backbone.downscale
        self.backbone = self.backbone.model

        # self.trainable_params['logit_scale'] = self.backbone.logit_scale
        # for param_key in self.trainable_params:
        #     self.trainable_params[param_key].requires_grad_(True)

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

        self.ce_fc = nn.Linear(self.nbit, self.nclass)

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

    def get_attn_maps(self, x):
        return self.forward_sem(x)

    def get_training_modules(self):
        return nn.ModuleDict({  # 'trainable_params': self.trainable_params,
            'ce_fc': self.ce_fc,
            'hash_fcs': self.hash_fcs,
            'sem_attns': self.sem_attns,
            'icons': self.icons
        })

    def forward(self, x, return_codes_and_attn_only=False):
        h, w = x.shape[-2:]
        x = self.backbone(x).last_hidden_state[:, 1:, :]
        # ntokens = int(x.size(1) ** 0.5)

        x = x.transpose(1, 2).reshape(x.size(0), x.size(2), h // self.downscale, w // self.downscale)

        attn_map = self.forward_sem(x)

        codes = []
        for i in range(self.nattns):
            attn = attn_map[:, i:i + 1, :, :]
            attn_x = x * attn
            local_feat = self.icons[i](attn_x)
            local_hash = self.hash_fcs[i](local_feat)
            codes.append(local_hash)

        global_feat = self.icons[-1](x)
        global_hash = self.hash_fcs[-1](global_feat)
        codes.append(global_hash)
        codes = torch.cat(codes, dim=1)
        logits = self.ce_fc(codes)

        if return_codes_and_attn_only:
            return codes, attn_map

        return codes, None, logits


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


class SEMICONCEWithAdapterWithFixedPrompt(SEMICONCEWithAdapter):

    def __init__(self, backbone: nn.Module, nbit: int, nclass: int, nattns: int = 4, **kwargs):
        super().__init__(backbone, nbit, nclass, nattns, **kwargs)

        self.ce_fc = TempCE(kwargs['fixed_center'], nbit, kwargs.get('temp', 10), kwargs.get('nonlinear', True))


class SEMICONCE(nn.Module):
    def __init__(self,
                 nbit: int,
                 nclass: int,
                 attn_size: int = 3,
                 feat_size: int = 2048,
                 has_global_codes: bool = True,
                 global_only: bool = False,
                 loss_method: str = 'ce',
                 **kwargs):
        super(SEMICONCE, self).__init__()

        pretrained = True
        code_length = nbit
        self.nbit = nbit
        self.nclass = nclass
        self.feat_size = feat_size
        self.attn_size = attn_size
        self.loss_method = loss_method
        self.has_global_codes = has_global_codes
        self.global_only = global_only

        self.backbone = SEMICON_backbone(pretrained=pretrained)

        if self.has_global_codes:
            self.refine_global = SEMICON_refine(is_local=False, pretrained=pretrained)

            if self.global_only:
                self.W_G = nn.Parameter(torch.Tensor(code_length, feat_size))
                torch.nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))
                self.W = nn.ParameterDict({'W_G': self.W_G})
            else:
                remaining = code_length - ((code_length // 2) + attn_size * (code_length // (2 * attn_size)))
                self.W_G = nn.Parameter(torch.Tensor(code_length // 2 + remaining, feat_size))
                torch.nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))
                self.W = nn.ParameterDict({'W_G': self.W_G})

                # local
                for i in range(attn_size):
                    W_Ln = nn.Parameter(torch.Tensor(code_length // (2 * attn_size), feat_size))
                    torch.nn.init.kaiming_uniform_(W_Ln, a=math.sqrt(5))
                    self.W[f'W_L{i}'] = W_Ln

                # self.W_L1 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
                # torch.nn.init.kaiming_uniform_(self.W_L1, a=math.sqrt(5))
                # self.W_L2 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
                # torch.nn.init.kaiming_uniform_(self.W_L2, a=math.sqrt(5))
                # self.W_L3 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
                # torch.nn.init.kaiming_uniform_(self.W_L3, a=math.sqrt(5))
        else:
            self.refine_global = None
            self.W = nn.ParameterDict()
            remaining = code_length - (attn_size * (code_length // attn_size))

            # local
            for i in range(attn_size):
                if i == attn_size - 1:
                    W_Ln = nn.Parameter(torch.Tensor(code_length // attn_size + remaining, feat_size))
                else:
                    W_Ln = nn.Parameter(torch.Tensor(code_length // attn_size, feat_size))
                torch.nn.init.kaiming_uniform_(W_Ln, a=math.sqrt(5))
                self.W[f'W_L{i}'] = W_Ln

        if not self.global_only:
            self.refine_local = SEMICON_refine(pretrained=pretrained)
            self.attention = SEMICON_attention(att_size=attn_size)
        else:
            self.refine_local = None
            self.attention = None

        self.hash_layer_active = nn.Sequential(
            nn.Tanh(),
        )
        self.code_length = code_length

        if loss_method == 'ce':
            self.ce_fc = nn.Linear(code_length, nclass)
        elif loss_method == 'cossim':
            self.ce_fc = CosSim(code_length, nclass)
        else:
            codebook = torch.randn(nclass, code_length).sign()
            self.ce_fc = CosSim(code_length, nclass, codebook, learn_cent=False)

        self.bernoulli = torch.distributions.Bernoulli(0.5)

    def get_backbone(self):
        return self.backbone

    def get_training_modules(self):
        return nn.ModuleDict({
            'refine_global': self.refine_global,
            'refine_local': self.refine_local,
            'attention': self.attention,
            'W': self.W,
            'ce': self.ce_fc
        })

    def count_parameters(self, mode='trainable'):
        if mode == 'trainable':
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        elif mode == 'non-trainable':
            return sum(p.numel() for p in self.parameters() if not p.requires_grad)
        else:  # all
            return sum(p.numel() for p in self.parameters())

    def forward(self, x, get_attention=False):
        out = self.backbone(x)  # .detach()
        if self.global_only:
            global_f = self.refine_global(out)
            deep_S = F.linear(global_f, self.W_G)

            ret = self.hash_layer_active(deep_S)
            logits = self.ce_fc(ret)
            return ret, None, logits
        else:
            batch_size, channels, h, w = out.shape
            att_map = self.attention(out)  # batchsize * att-size * 14 * 14
            att_size = att_map.shape[1]
            att_map_rep = att_map.unsqueeze(axis=2).repeat(1, 1, channels, 1, 1)
            out_rep = out.unsqueeze(axis=1).repeat(1, att_size, 1, 1, 1)
            out_local = att_map_rep.mul(out_rep)

            sub_codes = []
            avg_local_fs = []
            for i in range(att_size):
                out_local_sub = out_local[:, i].reshape(batch_size, channels, h, w)
                local_f, avg_local_f = self.refine_local(out_local_sub)
                sub_code = F.linear(avg_local_f, self.W[f'W_L{i}'])
                sub_codes.append(sub_code)
                avg_local_fs.append(avg_local_f)

            avg_local_fs = torch.stack(avg_local_fs, dim=1)

            if self.has_global_codes:
                global_f = self.refine_global(out)
                global_code = F.linear(global_f, self.W_G)

                sub_codes = torch.cat(sub_codes, dim=1)
                deep_S = torch.cat([global_code, sub_codes], dim=1)
            else:
                deep_S = sub_codes

            ret = self.hash_layer_active(deep_S)
            logits = self.ce_fc(ret)

            if get_attention:
                return ret, avg_local_fs, logits, att_map
            else:
                return ret, avg_local_fs, logits


class SEMICONCEWithFixedPrompt(SEMICONCE):

    def __init__(self, nbit: int, nclass: int, attn_size: int = 3, feat_size: int = 2048, has_global_codes: bool = True,
                 global_only: bool = False, loss_method: str = 'ce', **kwargs):
        super().__init__(nbit, nclass, attn_size, feat_size, has_global_codes, global_only, loss_method, **kwargs)

        self.ce_fc = TempCE(kwargs['fixed_center'], nbit, kwargs.get('temp', 10), kwargs.get('nonlinear', True))


if __name__ == '__main__':
    model = SEMICONCE(attn_size=1,
                      nbit=64,
                      nclass=10)
    print(model)

    print(model(torch.randn(1, 3, 224, 224)))
