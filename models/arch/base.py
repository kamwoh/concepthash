import torch.nn as nn
from transformers import ViTModel, CLIPModel, CLIPVisionModel

from models.layers.adapter import vit_add_adapter_, clip_add_adapter_, clip_add_attention_adapter_
from models.layers.lambda_layer import Lambda


class BaseNet(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 has_adapter: bool = False,
                 adapter_bottleneck_dim: float = 384,
                 adapter_dropout: float = 0.0,
                 adapter_mlp_1: bool = True,
                 adapter_mlp_2: bool = True,
                 attention_adapter: bool = False,
                 **kwargs):
        super().__init__()

        self.backbone = backbone
        self.nbit = nbit
        self.nclass = nclass

        self.has_adapter = has_adapter
        self.adapter_params = nn.ParameterDict()

        if self.has_adapter:
            if isinstance(self.backbone.model, ViTModel):
                vision_model = self.backbone.model.encoder
                vit_add_adapter_(vision_model, adapter_bottleneck_dim, self.adapter_params, adapter_dropout)
            elif isinstance(self.backbone.model, (CLIPModel, CLIPVisionModel)):
                if attention_adapter:
                    vision_model = self.backbone.model.vision_model
                    clip_add_attention_adapter_(vision_model, adapter_bottleneck_dim, self.adapter_params,
                                                dropout=adapter_dropout)
                else:
                    vision_model = self.backbone.model.vision_model
                    clip_add_adapter_(vision_model, adapter_bottleneck_dim, self.adapter_params,
                                      dropout=adapter_dropout,
                                      adapt_mlp_1=adapter_mlp_1, adapt_mlp_2=adapter_mlp_2)
            else:
                raise NotImplementedError

        # todo: set feature size in backbone
        if isinstance(self.backbone.model, ViTModel):
            self.backbone.model.pooler = Lambda(lambda x: x[:, 0])
            self.backbone.features_size = self.backbone.model.encoder.config.hidden_size
        elif isinstance(self.backbone.model, (CLIPModel, CLIPVisionModel)):
            self.backbone.features_size = self.backbone.model.vision_model.config.hidden_size

    def count_parameters(self, mode='trainable'):
        if mode == 'trainable':
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        elif mode == 'non-trainable':
            return sum(p.numel() for p in self.parameters() if not p.requires_grad)
        else:  # all
            return sum(p.numel() for p in self.parameters())

    def finetune_reset(self, *args, **kwargs):
        pass

    def get_backbone(self):
        return self.backbone

    def get_training_modules(self):
        return nn.ModuleDict()

    def get_adapter(self):
        return self.adapter_params

    def forward(self, *args, **kwargs):
        raise NotImplementedError('please implement `forward`')
