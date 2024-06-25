import torch
import torch.nn as nn


class SimLayer(nn.Module):
    def __init__(self, nfeat, nclass, codebook=None, learn_cent=True):
        super(SimLayer, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent

        if codebook is None:  # if no centroids, by default just usual weight
            codebook = torch.randn(nclass, nfeat)

        if not learn_cent:
            self.register_buffer('centroids', codebook.clone())
        else:
            if isinstance(codebook, nn.Parameter):
                self.centroids = codebook
            else:
                self.centroids = nn.Parameter(codebook.clone())

    def forward(self, x):
        nfeat = x
        ncenters = self.centroids

        dist = (nfeat.pow(2).sum(dim=1, keepdim=True) +
                ncenters.pow(2).sum(dim=1).unsqueeze(0) -
                2 * torch.matmul(nfeat, ncenters.t()))

        logits = - dist

        return logits

    def extra_repr(self) -> str:
        return 'in_features={}, n_class={}, learn_centroid={}'.format(
            self.nfeat, self.nclass, self.learn_cent
        )
