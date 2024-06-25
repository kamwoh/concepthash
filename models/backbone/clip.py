import math

import torch
import torch.nn as nn
from torchvision.models import resnet50
from transformers import CLIPModel, CLIPVisionModel, CLIPVisionConfig, CLIPConfig
from transformers.models.clip.modeling_clip import (
    CLIPTextTransformer,
    add_start_docstrings_to_model_forward,
    CLIP_TEXT_INPUTS_DOCSTRING,
    replace_return_docstrings,
    BaseModelOutputWithPooling,
    CLIPTextConfig,
    Optional,
    Tuple,
    Union,
    CLIPVisionEmbeddings
)

try:
    from transformers.models.clip.modeling_clip import (
        _create_4d_causal_attention_mask,
        _prepare_4d_attention_mask,
    )
except:
    from transformers.models.clip.modeling_clip import _expand_mask as _prepare_4d_attention_mask
    from transformers.models.clip.modeling_clip import _make_causal_mask as _create_4d_causal_attention_mask

from models.backbone.base import BaseNet


class CLIPWithR50(nn.Module):
    def __init__(self, name="openai/clip-vit-base-patch32", **kwargs):
        super().__init__()

        model = CLIPModel.from_pretrained(name)  # type: CLIPModel
        self.text_model = model.text_model
        self.vision_model = resnet50(pretrained=True)

        self.vision_dim = 2048
        self.text_dim = model.text_model.config.hidden_size
        self.projection_dim = model.vision_model.config.projection_dim

        self.text_projection = model.text_projection
        self.features_size = self.vision_dim

        del self.vision_model.fc
        del model.vision_model

    def resnet_forward(self, x):
        x = self.vision_model.conv1(x)
        x = self.vision_model.bn1(x)
        x = self.vision_model.relu(x)
        x = self.vision_model.maxpool(x)

        x = self.vision_model.layer1(x)
        x = self.vision_model.layer2(x)
        x = self.vision_model.layer3(x)
        x = self.vision_model.layer4(x)

        z = self.vision_model.avgpool(x)
        z = torch.flatten(z, 1)
        return x, z

    def forward(self, image):
        return self.resnet_forward(image)[1]


def _interpolate_pos_encoding(embeddings_module: CLIPVisionEmbeddings, x, w, h):
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


def _patch_forward_visual_embeddings(self, pixel_values):
    embeddings_module = self
    batch_size, _, w, h = pixel_values.shape
    patch_embeds = embeddings_module.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    class_embeds = embeddings_module.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    if embeddings.size(1) == embeddings_module.position_ids.size(1):
        embeddings = embeddings + embeddings_module.position_embedding(embeddings_module.position_ids)
    else:
        embeddings = embeddings + _interpolate_pos_encoding(embeddings_module, embeddings, w, h)
    return embeddings


class CLIP(nn.Module):

    def __init__(self, name="openai/clip-vit-base-patch32", **kwargs):
        super().__init__()

        cfg = CLIPConfig.from_pretrained(name)
        if kwargs.get('dropout', 0) != 0:
            cfg.vision_config.dropout = kwargs['dropout']
        if kwargs.get('attention_dropout', 0) != 0:
            cfg.vision_config.attention_dropout = kwargs['attention_dropout']

        self.model = CLIPModel.from_pretrained(name, config=cfg)  # type: CLIPModel
        self.downscale = 32 if '32' in name else 16  # if patch size = 32
        self.model.vision_model.embeddings.__class__.forward = _patch_forward_visual_embeddings

    def forward_feature_maps(self, x):
        h, w = x.shape[-2:]
        output = self.model.vision_model(x).last_hidden_state[:, 1:, :]
        # ntokens = output.size(1)
        # nsize = int(ntokens ** 0.5)
        # assert (nsize ** 2) == ntokens
        return output.permute(0, 2, 1).reshape(output.size(0),
                                               output.size(2),
                                               h // self.downscale,
                                               w // self.downscale)

    def forward(self, image):
        return self.model.vision_model(image).pooler_output


class ImageToTextTokenCLIPTextTransformer(CLIPTextTransformer):

    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            image_embeddings: Optional[torch.Tensor] = None  # image embeddings in token form
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ##### original input ids
        if input_ids is None and image_embeddings is None:
            raise ValueError("You have to specify either input_ids or image_embeddings")

        if image_embeddings is None:  # must check image embeddings first, since both will be passed in
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
            ##### end
        else:
            ##### image embeddings
            input_shape = image_embeddings.shape[:2]
            hidden_states = self.embeddings(inputs_embeds=image_embeddings, position_ids=position_ids)

        bsz, seq_len = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=input_ids.device), input_ids.to(torch.int).argmax(dim=-1)
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPVision(BaseNet):

    def __init__(self,
                 name="openai/clip-vit-base-patch32",
                 **kwargs):
        super().__init__()

        cfg = CLIPVisionConfig.from_pretrained(name)
        if kwargs.get('dropout', 0) != 0:
            cfg.dropout = kwargs['dropout']
        if kwargs.get('attention_dropout', 0) != 0:
            cfg.attention_dropout = kwargs['attention_dropout']

        self.model = CLIPVisionModel.from_pretrained(name, config=cfg)  # type: CLIPVisionModel
        self.features_size = self.model.config.hidden_size
        self.downscale = 32 if '32' in name else 16  # if patch size = 32
        self.model.vision_model.embeddings.__class__.forward = _patch_forward_visual_embeddings

    def forward_feature_maps(self, x):
        h, w = x.shape[-2:]
        output = self.model.vision_model(x).last_hidden_state[:, 1:, :]
        # ntokens = output.size(1)
        # nsize = int(ntokens ** 0.5)
        # assert (nsize ** 2) == ntokens
        return output.permute(0, 2, 1).reshape(output.size(0),
                                               output.size(2),
                                               h // self.downscale,
                                               w // self.downscale)

    def forward(self, x):
        output = self.model(x).pooler_output
        return output


if __name__ == '__main__':
    model = CLIP().cuda()
    for _ in range(100000):
        model(torch.randn(32, 3, 224, 224).cuda())
