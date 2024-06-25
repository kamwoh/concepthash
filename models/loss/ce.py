import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, multiclass=False, margin=0, scale=1, m_type='ce', **kwargs):
        super(CELoss, self).__init__()

        self.multiclass = multiclass
        self.m = margin
        self.s = scale
        self.m_type = m_type

        self.losses = {}

    def compute_margin_logits(self, logits, labels):
        if self.m_type == 'cos':
            if self.multiclass:
                y_onehot = labels * self.m
                margin_logits = self.s * (logits - y_onehot)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                margin_logits = self.s * (logits - y_onehot)
        else:
            if self.multiclass:
                y_onehot = labels * self.m
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits

        return margin_logits

    def forward(self, logits, code_logits, labels, onehot=True):
        """
        logits: for cross entropy
        code_logits: output from hash FC
        labels: labels
        onehot: whether labels is onehot
        """
        if self.m_type == 'ce':
            if self.multiclass:
                loss = F.binary_cross_entropy(torch.sigmoid(logits), labels.float())
            else:
                if onehot:
                    labels = labels.argmax(1)
                loss = F.cross_entropy(logits, labels)
        else:
            if not self.multiclass and (onehot or len(labels.size()) == 2):
                labels = labels.argmax(1)
            logits = self.compute_margin_logits(logits, labels)
            loss = F.cross_entropy(logits, labels)

        self.losses['ce'] = loss

        return loss
