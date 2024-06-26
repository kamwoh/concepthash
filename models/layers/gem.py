import torch.nn as nn


class GeM(nn.Module):
    """Generalized Mean Pooling.
    Paper: https://arxiv.org/pdf/1711.02512.
    """

    def __init__(self, p: int = 3, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(2, 3)).pow(1.0 / self.p)
