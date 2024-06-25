import math

import torch
import torch.nn as nn


class GaussianMixture(nn.Module):
    def __init__(self, nbit):
        super(GaussianMixture, self).__init__()
        self.nbit = nbit

        self.mean = nn.Parameter(torch.cat([torch.ones(self.nbit, 1) * -1,
                                            torch.ones(self.nbit, 1) * 1], dim=1))
        self.std = nn.Parameter(torch.ones(self.nbit, 2) * -1)
        self.prior = nn.Parameter(torch.ones(self.nbit, 2))

    def gaussian(self, x):
        # x = (N, nbit)
        eps = 1e-7
        mu = self.mean
        std = torch.exp(self.std)

        dist2mu = (x.unsqueeze(2) - mu.unsqueeze(0)).pow(2)  # (N, nbit, 2)

        numerator = torch.exp(- dist2mu / (2 * std.unsqueeze(0) + eps))
        denominator = torch.sqrt(2 * math.pi * std.unsqueeze(0)) + eps

        g = numerator / denominator
        return g

    def forward(self, x):
        prior = torch.exp(self.prior)
        prior = prior / prior.sum(dim=1, keepdim=True)

        return prior * self.gaussian(x)  # (N, nbit, 2)
