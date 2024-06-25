import math

import torch
import torch.nn as nn


class ToUniform(nn.Module):
    def forward(self, x):
        return torch.special.erf(x / math.sqrt(2))
