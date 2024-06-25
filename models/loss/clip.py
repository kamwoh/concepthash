import torch.nn as nn
from transformers.models.clip.modeling_clip import clip_loss


class CLIPLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CLIPLoss, self).__init__()

        self.losses = {}

    def forward(self, logits_per_text):
        loss = clip_loss(logits_per_text)
        return loss
