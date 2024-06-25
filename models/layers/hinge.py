import torch
import torch.nn as nn


class HingeSim(nn.Module):
    def __init__(self, nfeat, nclass, codebook=None, learn_cent=True, margin=1, act='relu'):
        super().__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.margin = margin
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

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'softplus':
            self.act = nn.Softplus()
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        x: (B, nfeat)
        """
        B, _ = x.size()

        w = x.reshape(B, 1, self.nfeat)
        y = self.centroids.reshape(1, self.nclass, self.nfeat)

        h = self.margin - w * y  # (B, nclass, nfeat)
        h = self.act(h).sum(dim=2)
        return h

    def extra_repr(self) -> str:
        return 'in_features={}, n_class={}, learn_centroid={}, margin={}'.format(
            self.nfeat, self.nclass, self.learn_cent, self.margin
        )
