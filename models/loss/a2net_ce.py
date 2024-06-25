import torch
import torch.nn.functional as F

from models.loss.base import BaseLoss


class A2NetCELoss(BaseLoss):

    def __init__(self, gamma=1, hash=1, decorr=0.1, **kwargs):
        super(A2NetCELoss, self).__init__()
        self.gamma = gamma
        self.hash = hash
        self.decorr = decorr

    def forward(self, codes, codes_tanh, logits, all_x, rec_all_x, labels):
        hash_loss = F.cross_entropy(logits, labels)

        corr = codes_tanh.t() @ codes_tanh  # (nbit, nbit)
        identity = torch.eye(codes_tanh.size(1), device=codes.device)
        decorr_loss = (corr - identity * codes_tanh.size(0)).pow(2).mean()
        rec_loss = F.mse_loss(rec_all_x, all_x.detach()) + self.gamma * F.mse_loss(codes, codes_tanh)

        self.losses['hash'] = hash_loss
        self.losses['decorr'] = decorr_loss
        self.losses['rec'] = rec_loss

        loss = self.hash * hash_loss + self.decorr * decorr_loss + rec_loss
        return loss
