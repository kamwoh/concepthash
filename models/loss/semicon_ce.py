import torch
import torch.nn.functional as F

from models.loss.base import BaseLoss


class SEMICONCELoss(BaseLoss):

    def __init__(self, gamma=0.1, loss_method='ce', **kwargs):
        super(SEMICONCELoss, self).__init__()
        self.gamma = gamma
        self.loss_method = loss_method

    def forward(self, codes, logits, labels):
        """

        :param codes: NxB
        :param logits: NxC
        :param labels: N
        :return:
        """
        if self.loss_method == 'ce':
            hash_loss = F.cross_entropy(logits, labels)
        else:
            scale = 8
            margin = 0.2
            y_onehot = torch.zeros_like(logits)
            y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), margin)
            margin_logits = scale * (logits - y_onehot)
            hash_loss = F.cross_entropy(margin_logits, labels)

        quantization_loss = torch.pow(codes - codes.sign(), 2).mean()

        self.losses['hash'] = hash_loss
        self.losses['quan'] = quantization_loss

        loss = hash_loss + quantization_loss * self.gamma
        return loss
