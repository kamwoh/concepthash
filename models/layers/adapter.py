import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPConfig
from transformers.models.vit.modeling_vit import ViTLayer, ViTConfig, ViTOutput


class Adapter(nn.Module):
    def __init__(
            self,
            in_dim,
            bottleneck_dim,
            dropout=0.0,
            adapter_scalar="learnable_scalar",
            adapter_layernorm_option="in",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim

        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm = nn.LayerNorm(self.in_dim)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.in_dim, self.bottleneck_dim)
        self.non_linear_func = nn.GELU()
        self.up_proj = nn.Linear(self.bottleneck_dim, self.in_dim)

        self.dropout = dropout
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm(up)

        output = up
        return output


class CLIPEncoderLayerWithVPT(CLIPEncoderLayer):
    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(config)

        self.config = config

        self.add_pe = kwargs.get('add_pe', False)
        self.ncontext = kwargs.get('ncontext', 8)
        self.num_tokens = kwargs.get('num_tokens', 50)

        if self.add_pe:
            self.pe = nn.Parameter(torch.randn(1, self.ncontext, config.hidden_size) * 0.02)
            self.register_buffer('zero_pe', torch.zeros(1, self.num_tokens, config.hidden_size))

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            causal_attention_mask: torch.Tensor,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        pe = torch.cat([self.zero_pe, self.pe], dim=1).repeat(hidden_states.size(0), 1, 1)
        hidden_states = hidden_states + pe
        return super().forward(hidden_states, attention_mask, causal_attention_mask, output_attentions)


class CLIPEncoderLayerWithAdapter(CLIPEncoderLayer):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.config = config
        self.adapt_mlp_1 = None
        self.adapt_mlp_2 = None

    def get_adapt_params(self):
        return list(nn.ModuleDict({
            'adapt_mlp_1': self.adapt_mlp_1,
            'adapt_mlp_2': self.adapt_mlp_2
        }).named_parameters())

    def setup_adapt_mlp(self, bottleneck_dim, dropout):
        if self.config.adapt_mlp_1:
            self.adapt_mlp_1 = Adapter(
                self.embed_dim,
                bottleneck_dim,
                dropout,
            )
        if self.config.adapt_mlp_2:
            self.adapt_mlp_2 = Adapter(
                self.embed_dim,
                bottleneck_dim,
                dropout,
            )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            causal_attention_mask: torch.Tensor,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

        if self.adapt_mlp_1 is not None:
            adapted_states = self.adapt_mlp_1(hidden_states)
        else:
            adapted_states = 0

        hidden_states = residual + hidden_states + adapted_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.adapt_mlp_2 is not None:
            adapted_states = self.adapt_mlp_2(hidden_states)
        else:
            adapted_states = 0

        hidden_states = residual + hidden_states + adapted_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPAttentionWithAdapter(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def set_num_parts(self, num_parts):
        self.num_parts = num_parts

    def setup_adapter(self, bottleneck_dim, dropout=0.0):
        self.q_adapter = Adapter(self.embed_dim,
                                 bottleneck_dim,
                                 dropout)
        # self.q_adapter = LoRALinearLayer(self.embed_dim, self.embed_dim, rank)
        self.k_adapter = Adapter(self.embed_dim,
                                 bottleneck_dim,
                                 dropout)
        # self.k_adapter = LoRALinearLayer(self.embed_dim, self.embed_dim, rank)
        self.v_adapter = Adapter(self.embed_dim,
                                 bottleneck_dim,
                                 dropout)
        # self.v_adapter = LoRALinearLayer(self.embed_dim, self.embed_dim, rank)
        self.out_adapter = Adapter(self.embed_dim,
                                   bottleneck_dim,
                                   dropout)
        # self.out_adapter = LoRALinearLayer(self.embed_dim, self.embed_dim, rank)
        # self.scale_adapter = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.embed_dim, bias=False),
        #     nn.GELU(),
        #     nn.Linear(self.embed_dim, 1, bias=False),
        #     nn.Sigmoid()
        # )

    def get_adapt_params(self):
        return list(nn.ModuleDict({
            'q_adapter': self.q_adapter,
            'k_adapter': self.k_adapter,
            'v_adapter': self.v_adapter,
            'out_adapter': self.out_adapter,
            # 'scale_adapter': self.scale_adapter
        }).named_parameters())

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        scale = self.scale
        query_states = (self.q_proj(hidden_states) + self.q_adapter(hidden_states)) * scale
        key_states = self._shape((self.k_proj(hidden_states) + self.k_adapter(hidden_states)), -1, bsz)
        value_states = self._shape((self.v_proj(hidden_states) + self.v_adapter(hidden_states)), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # (B, N, HW, HW)

        # attn_mean = attn_weights.mean(dim=-1)  # (B, N, 1 + HW + num_parts)
        # attn_mask = torch.ones_like(attn_mean)
        # attn_image_means = attn_mean[..., 1:-self.num_parts]  # (B, N, HW)
        # attn_mask[..., 1:-self.num_parts] = attn_image_means >= attn_image_means.mean(dim=-1, keepdim=True)
        # attn_mask = attn_mask.unsqueeze(-1).float()  # (B, N, 1 + HW + num_parts, 1)
        # attn_weights = attn_mask * attn_weights + (1 - attn_mask) * -1e6

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output) + self.out_adapter(attn_output)

        return attn_output, attn_weights_reshaped


class VitOutputWithAdapter(ViTOutput):
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, adapter_layer=None) -> torch.Tensor:
        assert adapter_layer is not None

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        adapted_states = adapter_layer(hidden_states)

        hidden_states = hidden_states + input_tensor + adapted_states

        return hidden_states


class ViTLayerWithAdapter(ViTLayer):
    def __init__(self, config: ViTConfig):
        super().__init__(config)

        self.config = config
        self.output = VitOutputWithAdapter(config)
        self.adapt_mlp_1 = None
        self.adapt_mlp_2 = None

    def get_adapt_params(self):
        return list(nn.ModuleDict({
            'adapt_mlp_1': self.adapt_mlp_1,
            'adapt_mlp_2': self.adapt_mlp_2
        }).named_parameters())

    def setup_adapt_mlp(self, bottleneck_dim, dropout):
        self.adapt_mlp_1 = Adapter(
            self.config.hidden_size,
            bottleneck_dim,
            dropout,
        )
        self.adapt_mlp_2 = Adapter(
            self.config.hidden_size,
            bottleneck_dim,
            dropout,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        adapted_states = self.adapt_mlp_1(attention_output)

        # first residual connection
        hidden_states = attention_output + hidden_states + adapted_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states, self.adapt_mlp_2)

        outputs = (layer_output,) + outputs

        return outputs


def clip_add_adapter_(vision_model, adapter_bottleneck_dim, trainable_params=None,
                      adapt_mlp_1=True, adapt_mlp_2=True, dropout=0.0):
    if trainable_params is None:
        trainable_params = nn.ParameterDict()
    current_layers = vision_model.encoder.layers

    clip_encoder_cfg = vision_model.encoder.config
    clip_encoder_cfg.adapt_mlp_1 = adapt_mlp_1
    clip_encoder_cfg.adapt_mlp_2 = adapt_mlp_2
    new_layers = nn.ModuleList([CLIPEncoderLayerWithAdapter(clip_encoder_cfg) for _ in current_layers])
    for i, (nlayer, clayer) in enumerate(zip(new_layers, current_layers)):  # type: CLIPEncoderLayerWithAdapter
        nlayer.load_state_dict(clayer.state_dict())
        nlayer.setup_adapt_mlp(adapter_bottleneck_dim, dropout)
        nlayer_named_params = nlayer.get_adapt_params()
        for pname, param in nlayer_named_params:
            trainable_params[f'adapter_{i}_{pname.replace(".", "_")}'] = param
            param.requires_grad_(True)
    vision_model.encoder.layers = new_layers
    return trainable_params


def clip_add_attention_adapter_(vision_model, adapter_bottleneck_dim, trainable_params=None, dropout=0.0):
    clip_encoder_cfg = vision_model.encoder.config
    current_layers = vision_model.encoder.layers

    for i, clayer in enumerate(current_layers):
        self_attn_sd = clayer.self_attn.state_dict()
        clayer.self_attn = CLIPAttentionWithAdapter(clip_encoder_cfg)
        clayer.self_attn.load_state_dict(self_attn_sd)
        clayer.self_attn.setup_adapter(adapter_bottleneck_dim)
        clayer_named_params = clayer.self_attn.get_adapt_params()
        for pname, param in clayer_named_params:
            trainable_params[f'adapter_{i}_{pname.replace(".", "_")}'] = param
            param.requires_grad_(True)

    return trainable_params


def clip_add_myvpt_(vision_model, ncontext, num_tokens, trainable_params=None):
    if trainable_params is None:
        trainable_params = nn.ParameterDict()
    current_layers = vision_model.encoder.layers

    clip_encoder_cfg = vision_model.encoder.config
    new_layers = nn.ModuleList([CLIPEncoderLayerWithVPT(clip_encoder_cfg,
                                                        add_pe=True,
                                                        ncontext=ncontext,
                                                        num_tokens=num_tokens) for _ in current_layers])
    for i, (nlayer, clayer) in enumerate(zip(new_layers, current_layers)):  # type: CLIPEncoderLayerWithVPT
        print(nlayer.load_state_dict(clayer.state_dict(), strict=False))
        trainable_params[f'myvpt_{i}_pe'] = nlayer.pe
    vision_model.encoder.layers = new_layers
    return trainable_params


def vit_add_adapter_(vision_model, adapter_bottleneck_dim, trainable_params=None, dropout=0.0):
    if trainable_params is None:
        trainable_params = nn.ParameterDict()
    current_layers = vision_model.layer

    new_layers = nn.ModuleList([ViTLayerWithAdapter(vision_model.config) for _ in current_layers])
    for i, (nlayer, clayer) in enumerate(zip(new_layers, current_layers)):  # type: ViTLayerWithAdapter
        nlayer.load_state_dict(clayer.state_dict())
        nlayer.setup_adapt_mlp(adapter_bottleneck_dim, dropout)
        nlayer_named_params = nlayer.get_adapt_params()
        for pname, param in nlayer_named_params:
            trainable_params[f'adapter_{i}_{pname.replace(".", "_")}'] = param
            param.requires_grad_(True)
    vision_model.layer = new_layers
    return trainable_params
