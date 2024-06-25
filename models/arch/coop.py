import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from omegaconf import DictConfig
from transformers.models.clip import CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPVisionEmbeddings

from models.arch.base import BaseNet
from models.backbone.clip import CLIP, CLIPModel
from models.layers.adapter import clip_add_myvpt_
from models.layers.cossim import CosSim
from models.layers.iternorm import DBN


class SelfAttention(nn.Module):
    def __init__(self, params=True, dim=768, mask_sigma=0, ncontext=8,
                 cross_attention=False, differentiable=False,
                 strong=False, add_pe=False, num_tokens=50):
        super().__init__()

        if params:
            if strong:
                self.q = nn.Sequential(
                    nn.Linear(dim, dim, bias=False),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                    nn.Linear(dim, dim, bias=False)
                )
                self.k = nn.Sequential(
                    nn.Linear(dim, dim, bias=False),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                    nn.Linear(dim, dim, bias=False)
                )
                self.v = nn.Sequential(
                    nn.Linear(dim, dim, bias=False),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                    nn.Linear(dim, dim, bias=False)
                )
            else:
                self.q = nn.Linear(dim, dim, bias=False)
                self.k = nn.Linear(dim, dim, bias=False)
                self.v = nn.Linear(dim, dim, bias=False)
        else:
            self.q = nn.Identity()
            self.k = nn.Identity()
            self.v = nn.Identity()

        self.scale = dim ** -0.5
        self.mask_sigma = mask_sigma
        self.ncontext = ncontext
        self.cross_attention = cross_attention
        self.differentiable = differentiable
        self.add_pe = add_pe

        if self.add_pe:
            self.pe = nn.Parameter(torch.randn(1, self.ncontext, dim))
            self.register_buffer('zero_pe', torch.zeros(1, num_tokens, dim))
            self.num_tokens = num_tokens

    @staticmethod
    def GaussianKernel(radius, std):
        """
        Generate a gaussian blur kernel based on the given radius and std.
        Args
        ----------
        radius: int
            Radius of the Gaussian kernel. Center not included.
        std: float
            Standard deviation of the Gaussian kernel.
        Returns
        ----------
        weight: torch.FloatTensor, [2 * radius + 1, 2 * radius + 1]
            Output Gaussian kernel.
        """
        size = 2 * radius + 1
        weight = torch.ones(size, size)
        weight.requires_grad = False
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                dis = (i * i) + (j * j)
                weight[i + radius][j + radius] = np.exp(-dis / (2 * std * std))
        weight = weight / weight.sum()
        return weight

    def gaussian_mask(self):
        pass

    def mask_attention_map(self, attention_map, sigma=0.5, cache_gaussian=True):
        is_sequence = len(attention_map.size()) == 3
        if is_sequence:  # assume num_head = 1
            B, _, HW = attention_map.size()
            H = W = int(HW ** 0.5)
            attention_map = attention_map.reshape(B, -1, H, W)

        # Find the location with the max value in the attention map
        B, C, H, W = attention_map.size()

        if not self.differentiable:
            max_loc = torch.argmax(attention_map.reshape(B, C, -1), dim=2)
            max_y, max_x = torch.div(max_loc, H), torch.remainder(max_loc, H)
        else:
            max_y = torch.sum(attention_map.reshape(B, C, -1) * torch.arange(H, device=attention_map.device,
                                                                             dtype=torch.float32).view(1, 1, -1), dim=2,
                              keepdim=True)
            max_x = torch.sum(attention_map.reshape(B, C, -1) * torch.arange(W, device=attention_map.device,
                                                                             dtype=torch.float32).view(1, 1, -1), dim=2,
                              keepdim=True)

        # Create a 2D Gaussian kernel centered at (max_x, max_y)
        if cache_gaussian and hasattr(self, 'xx'):
            xx = self.xx
            yy = self.yy
        else:
            x = torch.arange(W, device=attention_map.device, dtype=torch.float32)
            y = torch.arange(H, device=attention_map.device, dtype=torch.float32)
            xx, yy = torch.meshgrid(x, y)

            xx = xx.reshape(1, 1, H, W)
            yy = yy.reshape(1, 1, H, W)
            self.xx = xx
            self.yy = yy

        max_x = max_x.reshape(B, C, 1, 1)
        max_y = max_y.reshape(B, C, 1, 1)
        gaussian_kernel = torch.exp(-((xx - max_x) ** 2 + (yy - max_y) ** 2) / (2 * sigma ** 2))

        # Normalize the Gaussian kernel
        gaussian_kernel /= torch.max(gaussian_kernel.reshape(B, C, -1), dim=2)[0].unsqueeze(2).unsqueeze(3)

        # Mask the attention map with the Gaussian kernel
        masked_attention_map = attention_map * gaussian_kernel  # (B, C, H, W)

        if is_sequence:
            masked_attention_map = masked_attention_map.reshape(B, C, -1)

        return masked_attention_map

    def forward(self, x):
        if self.add_pe:
            pe = torch.cat([self.zero_pe, self.pe], dim=1).repeat(x.size(0), 1, 1)
            x = x + pe

        q = k = v = x

        q = self.q(q)  # (B, L, D)
        k = self.k(k)
        v = self.v(v)

        attn = q @ k.transpose(1, 2)  # (B, Q, K)

        if self.cross_attention:
            # cross attn masking
            attn_region = torch.softmax(attn[:, -self.ncontext:, 1:-self.ncontext] * self.scale, dim=-1)
            if self.mask_sigma != 0:
                attn_region = self.mask_attention_map(attn_region, self.mask_sigma, True)
            attn[:, -self.ncontext:, 1:-self.ncontext] = attn_region

            mask = torch.zeros_like(attn)
            mask[:, -self.ncontext:, 1:-self.ncontext] = 1
            attn = mask * attn
        else:
            if self.mask_sigma != 0:
                attn_region = self.mask_attention_map(attn[:, -self.ncontext:, 1:-self.ncontext], self.mask_sigma, True)
                attn[:, -self.ncontext:, 1:-self.ncontext] = attn_region

            attn = torch.softmax(attn * self.scale, dim=-1)

        v = attn @ v  # (B, L, D)

        return attn.unsqueeze(1), v  # (B, 1, L, L), (B, L, D)


class LGHWithoutText(BaseNet):
    def __init__(self,
                 backbone: CLIP,
                 nbit: int,
                 nclass: int,
                 ncontext: int,
                 add_bn: bool = False,
                 use_before_projection: bool = True,
                 upt_config: DictConfig = None,
                 fixed_center: torch.Tensor = None,
                 additional_blocks: int = 0,
                 self_attn_at_last: nn.Module = nn.Identity(),
                 hash_head: nn.Module = nn.Identity(),
                 concept_reg: bool = False,
                 concept_cossim: bool = True,
                 concept_share_pe: bool = False,
                 vpt_pe: bool = False,
                 fixed_pe: bool = False,
                 nregs: int = 0,
                 hash_fc_nlayers: int = 1,
                 del_text_model: bool = True,
                 **kwargs):
        # assert isinstance(backbone, CLIP), 'only support CLIP for now'
        super().__init__(backbone, nbit, nclass, **kwargs)
        self.trainable_params = nn.ParameterDict()

        if vpt_pe:
            clip_add_myvpt_(self.backbone.model.vision_model, ncontext, 50, self.trainable_params)

        self.features_size = backbone.features_size
        self.backbone = backbone.model  # type: CLIPModel
        self.add_bn = add_bn
        self.use_before_projection = use_before_projection
        self.upt_config = upt_config
        self.ncontext = ncontext
        self.fixed_center = fixed_center
        self.additional_blocks = additional_blocks
        self.self_attn_at_last = self_attn_at_last
        self.hash_head = hash_head
        self.concept_reg = concept_reg
        self.concept_cossim = concept_cossim
        self.concept_share_pe = concept_share_pe
        self.fixed_pe = fixed_pe
        self.nregs = nregs
        self.hash_fc_nlayers = hash_fc_nlayers

        # caching #
        self.attn_cache = None

        vision_dim = self.backbone.vision_model.config.hidden_size
        projection_dim = self.backbone.vision_model.config.projection_dim

        self.vision_dim = vision_dim
        self.embed_dim = projection_dim

        #### for UPT ####
        self.hash_initialization()

        if self.additional_blocks > 0:
            for _ in range(self.additional_blocks):
                new_layer = CLIPEncoderLayer(self.backbone.vision_model.config)
                self.backbone.vision_model.encoder.layers.append(new_layer)
                for name, param in new_layer.named_parameters():
                    self.trainable_params['new_' + name.replace(".", "_")] = param

        if del_text_model:
            del self.backbone.text_model  # del to save memory

        if self.concept_reg:
            self.concept_initialization()

    def concept_initialization(self):
        if self.concept_share_pe:
            self.concept_pe = self.hash_pe
        else:
            if self.fixed_pe:
                concept_pe = torch.randn(1, self.ncontext, self.vision_dim)  # * 0.02
                self.register_buffer('concept_pe', concept_pe)
            else:
                self.concept_pe = nn.Parameter(torch.randn(1, self.ncontext, self.vision_dim) * 0.02)
                self.trainable_params['concept_pe'] = self.concept_pe

        if self.concept_cossim:
            self.concept_ce = CosSim(self.vision_dim, self.nclass)
            self.trainable_params['concept_ce_centroids'] = self.concept_ce.centroids
        else:
            self.concept_ce = nn.Linear(self.vision_dim, self.nclass, bias=False)
            self.trainable_params['concept_ce_weight'] = self.concept_ce.weight

    def forward_concept(self, concept_features):
        # concept_features: (B, Q, D)
        B, Q, D = concept_features.size()
        concept_features_pe = concept_features + self.concept_pe.to(concept_features).repeat(concept_features.size(0),
                                                                                             1,
                                                                                             1)
        concept_logits = self.concept_ce(concept_features_pe.reshape(B * Q, D)).reshape(B, Q, -1)
        return concept_logits.transpose(0, 1)  # (B, Q, C) -> (Q, B, C)

    def hash_initialization(self):
        ncontext = self.ncontext
        nregs = self.nregs
        nbit = self.nbit

        if self.upt_config is not None and self.upt_config.multi:
            self.multi = True

            if self.upt_config.get('single_hash_fc'):
                if not self.use_before_projection:
                    in_dim = self.embed_dim
                else:
                    in_dim = self.vision_dim

                if self.upt_config.get('ensemble_method', 'concat') == 'concat':
                    out_dim = nbit // ncontext
                else:
                    out_dim = nbit

                bit_dim = nbit

                if self.upt_config.get('hash_pe', False):
                    if self.fixed_pe:
                        hash_pe = torch.randn(1, ncontext, in_dim)
                        self.register_buffer('hash_pe', hash_pe)
                    else:
                        self.hash_pe = nn.Parameter(torch.randn(1, ncontext, in_dim))
                        self.trainable_params['hash_pe'] = self.hash_pe
                else:
                    self.register_buffer('hash_pe', torch.zeros(1, ncontext, in_dim))

                if self.hash_fc_nlayers == 1:
                    self.hash_fc = nn.Linear(in_dim, out_dim, bias=False)
                else:
                    layers = []
                    for _ in range(self.hash_fc_nlayers - 1):
                        layers.append(nn.Linear(in_dim, in_dim))
                        layers.append(nn.ReLU())
                    layers.append(nn.Linear(in_dim, out_dim, bias=False))
                    self.hash_fc = nn.Sequential(*layers)
            else:
                if not self.use_before_projection:
                    in_dim = self.embed_dim
                else:
                    in_dim = self.vision_dim

                if self.upt_config.get('ensemble_method', 'concat') == 'concat':
                    out_dim = nbit
                else:
                    out_dim = nbit * ncontext

                bit_dim = nbit

                if self.hash_fc_nlayers == 1:
                    self.hash_fc = nn.Conv1d(in_dim * ncontext, out_dim, 1, groups=ncontext, bias=False)
                else:
                    layers = []
                    for _ in range(self.hash_fc_nlayers - 1):
                        layers.append(nn.Conv1d(in_dim * ncontext, in_dim * ncontext, 1, groups=ncontext))
                        layers.append(nn.ReLU())
                    layers.append(nn.Conv1d(in_dim * ncontext, out_dim, 1, groups=ncontext, bias=False))
                    self.hash_fc = nn.Sequential(*layers)

            if self.add_bn:
                if self.add_bn == 'dbn':
                    self.hash_bn = DBN(bit_dim, ncontext, dim=2)
                else:
                    self.hash_bn = nn.BatchNorm1d(bit_dim)
            else:
                self.hash_bn = nn.Identity()

            if self.upt_config.get('upt_context', True):
                self.hash_queries = nn.Parameter(torch.randn(1, ncontext + nregs, self.embed_dim))
                self.hash_attention = nn.Module()
                self.hash_attention.sa = nn.MultiheadAttention(self.embed_dim, self.upt_config.num_heads,
                                                               batch_first=True, dropout=self.upt_config.dropout)
                self.hash_attention.ffn = nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.Dropout(self.upt_config.dropout),
                    nn.Linear(self.embed_dim, self.embed_dim)
                )
                self.hash_attention.norm1 = nn.LayerNorm(self.embed_dim)
                self.hash_attention.norm2 = nn.LayerNorm(self.embed_dim)
                self.hash_attention.ffn2 = nn.Linear(self.embed_dim, self.vision_dim)
                self.trainable_params['hash_queries'] = self.hash_queries
            else:
                self.visual_context = nn.Parameter(torch.randn(1, ncontext + nregs, self.vision_dim))
                self.hash_attention = None
                self.trainable_params['visual_context'] = self.visual_context
        else:
            self.multi = False

            #### hash fc ####
            if not self.use_before_projection:
                self.hash_fc = nn.Linear(self.embed_dim, nbit, bias=False)
                if self.add_bn:
                    if self.add_bn == 'dbn':
                        self.hash_bn = DBN(nbit, self.ncontext, dim=2)
                    else:
                        self.hash_bn = nn.BatchNorm1d(nbit)
                else:
                    self.hash_bn = nn.Identity()
            else:
                self.hash_fc = nn.Linear(self.vision_dim, nbit, bias=False)
                if self.add_bn:
                    if self.add_bn == 'dbn':
                        self.hash_bn = DBN(nbit, self.ncontext, dim=2)
                    else:
                        self.hash_bn = nn.BatchNorm1d(nbit)
                else:
                    self.hash_bn = nn.Identity()

        if self.fixed_center is not None:
            self.register_buffer('center', self.fixed_center)
        else:
            self.center = nn.Parameter(torch.randn(self.nclass, self.nbit) * 0.02)
            self.trainable_params['center'] = self.center

    def get_center(self):
        return self.center

    def get_backbone(self):
        return self.backbone.vision_model

    def get_training_modules(self):
        return nn.ModuleDict({
            'trainable_params': self.trainable_params,
            'hash_fc': self.hash_fc,
            'hash_bn': self.hash_bn,
            'hash_attention': self.hash_attention if self.multi else None,
            'self_attn_at_last': self.self_attn_at_last,
            'hash_head': self.hash_head
        })

    def forward_hash_query(self):  # UPT
        if self.upt_config.get('upt_context', True):
            if self.upt_config.get('v2'):
                x = self.hash_queries
                x = self.hash_attention.norm1(x + self.hash_attention.sa(x, x, x)[0])
                x = self.hash_attention.norm2(x + self.hash_attention.ffn(x))
                x = self.hash_attention.ffn2(x)
            else:
                x = self.hash_queries
                x = self.hash_attention.norm1(x) + self.hash_attention.sa(x, x, x)[0]
                x = self.hash_attention.norm2(x) + self.hash_attention.ffn(x)
                x = self.hash_attention.ffn2(x)
            return x
        else:
            return self.visual_context

    def interpolate_pos_encoding(self, x, w, h):
        embeddings_module: CLIPVisionEmbeddings = self.backbone.vision_model.embeddings
        npatch = x.shape[1] - 1
        N = embeddings_module.position_embedding.weight.shape[0] - 1
        if npatch == N and w == h:
            return embeddings_module.position_embedding.weight
        class_pos_embed = embeddings_module.position_embedding.weight[0]
        patch_pos_embed = embeddings_module.position_embedding.weight[1:]
        dim = x.shape[-1]
        w0 = w // embeddings_module.patch_size
        h0 = h // embeddings_module.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.reshape(1, 1, -1), patch_pos_embed), dim=1)

    def forward_visual_embeddings(self, pixel_values):
        embeddings_module = self.backbone.vision_model.embeddings
        batch_size, _, w, h = pixel_values.shape
        patch_embeds = embeddings_module.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = embeddings_module.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if embeddings.size(1) == embeddings_module.position_ids.size(1):
            embeddings = embeddings + embeddings_module.position_embedding(embeddings_module.position_ids)
        else:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, w, h)
        if self.upt_config.get('exclude_cls'):
            embeddings = embeddings[:, 1:, :]
        return embeddings

    def forward_visual(self, image_input, hash_queries=None, cache_attn=False):
        hidden_states = self.forward_visual_embeddings(image_input)
        if hash_queries is not None:
            hidden_states = torch.cat([hidden_states, hash_queries.repeat(hidden_states.size(0), 1, 1)], dim=1)
        hidden_states = self.backbone.vision_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.backbone.vision_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

        if cache_attn:
            self.attn_cache = encoder_outputs.attentions

        last_hidden_state = encoder_outputs.last_hidden_state
        hidden_states = encoder_outputs.hidden_states

        if not isinstance(self.self_attn_at_last, nn.Identity):
            last_attn, last_hidden_state = self.self_attn_at_last(last_hidden_state)

            if cache_attn:
                self.attn_cache = self.attn_cache + (last_attn,)

            hidden_states = hidden_states + (last_hidden_state,)

        if self.upt_config.get('exclude_cls'):
            pooled_output = last_hidden_state.mean(dim=1)
        else:
            pooled_output = last_hidden_state[:, 0, :]

        pooled_output = self.backbone.vision_model.post_layernorm(pooled_output)
        pooled_output = self.backbone.visual_projection(pooled_output)

        if hash_queries is not None:
            start = self.ncontext + self.nregs
            end = self.nregs
            if end == 0:
                hash_outputs = last_hidden_state[:, -start:, :]
            else:
                hash_outputs = last_hidden_state[:, -start:-end, :]
            if not self.use_before_projection:
                hash_outputs = self.backbone.vision_model.post_layernorm(hash_outputs)
                hash_outputs = self.backbone.visual_projection(hash_outputs)
            if not isinstance(self.hash_head, nn.Identity):
                hash_outputs = self.backbone.vision_model.post_layernorm(hash_outputs)
                hash_outputs = self.hash_head(hash_outputs)
        else:
            if not self.use_before_projection:
                hash_outputs = pooled_output
            else:
                hash_outputs = last_hidden_state[:, 0, :]

        return hidden_states, pooled_output, hash_outputs

    def forward(self, x, y=None, cache=False, update_cache=False):
        """

        :param x: image
        :return:
            u = logits
            x = code logits
        """
        ncontext = self.ncontext
        #### forward visual ####
        if self.multi:
            vision_context = self.forward_hash_query()
            image_hidden_states, image_features, hash_features = self.forward_visual(x, vision_context, True)
        else:
            image_hidden_states, image_features, hash_features = self.forward_visual(x)

        # if cache and not hasattr(self, 'caches'):
        #     self.caches = {}

        #### forward hash ####
        if self.multi:
            if self.upt_config.get('single_hash_fc'):
                # (B, Q, D) -> (B, Q, nbit // Q) -> (B, Q * nbit // Q)
                if self.upt_config.get('hash_pe', False):
                    hash_pe = self.hash_pe.to(x).repeat(x.size(0), 1, 1)
                    vision_hash = self.hash_fc(hash_features + hash_pe)
                else:
                    vision_hash = self.hash_fc(hash_features)

                if self.upt_config.get('ensemble_method', 'concat') == 'concat':
                    vision_hash = vision_hash.reshape(x.size(0), -1)
                else:
                    ens_vision_hash = vision_hash
                    vision_hash = vision_hash.mean(dim=1)

                vision_hash = self.hash_bn(vision_hash)  # (B, nbit)
            else:
                # (B, Q, D) -> (B, Q * D, 1) -> (B, Q * nbit // Q, 1)
                vision_hash = self.hash_fc(hash_features.reshape(x.size(0), -1, 1)).squeeze(2)

                if self.upt_config.get('ensemble_method', 'concat') == 'avg':
                    ens_vision_hash = vision_hash.reshape(x.size(0), ncontext, -1)
                    vision_hash = vision_hash.reshape(x.size(0), ncontext, -1).mean(dim=1)

                vision_hash = self.hash_bn(vision_hash)  # (B, nbit)
        else:
            vision_hash = self.hash_fc(hash_features)  # (B, nbit)
            vision_hash = self.hash_bn(vision_hash)  # (B, nbit)

        hash_center = self.get_center()

        # hash information
        vision_hash_l2 = F.normalize(vision_hash, dim=-1, p=2)
        hash_center_l2 = F.normalize(hash_center, dim=-1, p=2)

        cont_logits = vision_hash_l2 @ hash_center_l2.t()
        bin_logits = vision_hash_l2 @ (hash_center_l2.sign() / (self.nbit ** 0.5)).t()

        outputs = {
            'logits_cont': cont_logits,
            'logits_bin': bin_logits,
            'codes': vision_hash,
            'image_hidden_states': image_hidden_states,
            'hash_features': hash_features,
            'attn_cache': self.attn_cache
        }
        del self.attn_cache

        if self.multi and self.upt_config.get('ensemble_method', 'concat') == 'avg':
            outputs['ensemble_codes'] = ens_vision_hash

        if self.concept_reg:
            outputs['logits_concept'] = self.forward_concept(hash_features)

        return image_features, outputs


class LGHWithFixedPrompt(LGHWithoutText):

    def __init__(self, backbone: CLIP, nbit: int, nclass: int, ncontext: int, add_bn: bool = False,
                 use_before_projection: bool = True, upt_config: DictConfig = None, fixed_center: torch.Tensor = None,
                 additional_blocks: int = 0, text_projection: nn.Module = None, **kwargs):
        super().__init__(backbone, nbit, nclass, ncontext, add_bn, use_before_projection, upt_config, fixed_center,
                         additional_blocks, **kwargs)
        if text_projection is None:
            self.text_projection = nn.Linear(fixed_center.size(1), nbit)
        else:
            self.text_projection = text_projection

    def get_training_modules(self):
        return nn.ModuleDict({
            'trainable_params': self.trainable_params,
            'hash_fc': self.hash_fc,
            'hash_bn': self.hash_bn,
            'hash_attention': self.hash_attention if self.multi else None,
            'text_projection': self.text_projection,
            'self_attn_at_last': self.self_attn_at_last,
            'hash_head': self.hash_head
        })

    def get_center(self):
        return self.text_projection(self.center)


class LGHWithFixedPromptFILIP(LGHWithFixedPrompt):

    def __init__(self, backbone: CLIP, nbit: int, nclass: int, ncontext: int, add_bn: bool = False,
                 use_before_projection: bool = True, upt_config: DictConfig = None, fixed_center: torch.Tensor = None,
                 additional_blocks: int = 0, prompt_path='', processor: CLIPProcessor = None, **kwargs):
        super().__init__(backbone, nbit, nclass, ncontext, add_bn, use_before_projection, upt_config, fixed_center,
                         additional_blocks, del_text_model=False, **kwargs)

        prompts = open(prompt_path).readlines()
        prompts = [line.strip() for line in prompts]

        tokenized_prompts = processor(text=prompts,
                                      return_tensors="pt",
                                      padding=True,
                                      truncation=True,
                                      max_length=77)['input_ids']
        logging.info(f'Prompts: {tokenized_prompts.size()}')
        with torch.no_grad():
            token_embeds = self.backbone.text_model(tokenized_prompts)[0]  # (200, nT, d)
            token_embeds = self.backbone.text_projection(token_embeds)  # (200, nT, vd)

        self.register_buffer('token_embeds', token_embeds)

        del self.backbone.text_model
        del self.backbone.text_projection

    def forward(self, *args, **kwargs):
        image_features, outputs = super().forward(*args, **kwargs)

        hash_features = outputs['hash_features']  # (B, N, vd)
        if self.use_before_projection:
            hash_features = self.backbone.vision_model.post_layernorm(hash_features)
            hash_features = self.backbone.visual_projection(hash_features)  # (B, N, vd)
        token_features = self.token_embeds

        hash_features = hash_features / hash_features.norm(dim=-1, keepdim=True)  # (B, N, 512)
        token_features = token_features / token_features.norm(dim=-1, keepdim=True)  # (200, T, 512)

        # (B, 1, N, 512) @ (1, 200, 512, T) -> (B, 200, N, T)
        logits = hash_features.unsqueeze(1) @ token_features.transpose(1, 2).unsqueeze(0)
        logits_i2t = logits.max(dim=-1)[0]  # max text tokens
        logits_i2t = logits_i2t.mean(dim=-1)  # mean image tokens

        logits_t2i = logits.max(dim=-2)[0]  # max image tokens
        logits_t2i = logits_t2i.mean(dim=-1)  # mean text tokens

        outputs['logits_filip'] = (logits_i2t + logits_t2i) * 0.5
        outputs['logits_filip_t2i'] = logits_t2i
        outputs['logits_filip_i2t'] = logits_i2t

        return image_features, outputs
