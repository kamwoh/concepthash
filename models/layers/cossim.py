import torch
import torch.nn as nn
import torch.nn.functional as F


class CosSim(nn.Module):
    def __init__(self,
                 nfeat,
                 nclass,
                 codebook=None,
                 learn_cent=True,
                 signhash=None,
                 group=1,
                 single_quan=False,
                 input_group=1):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent
        self.group = group
        self.single_quan = single_quan
        self.input_group = input_group

        if codebook is None:  # if no centroids, by default just usual weight
            codebook = torch.randn(nclass, nfeat)

        if not learn_cent:
            self.register_buffer('centroids', codebook.clone())
        else:
            if isinstance(codebook, nn.Parameter):
                self.centroids = codebook
            else:
                self.centroids = nn.Parameter(codebook.clone())

        self.signhash = signhash

    def forward(self, x, sign_centroids=False, centroids=None):
        if centroids is None:
            centroids = self.centroids

        if self.signhash is not None:
            centroids = self.signhash(centroids)

        if sign_centroids:
            centroids = torch.sign(centroids)

        if self.single_quan:
            centroids_group = centroids.reshape(self.nclass, self.group, -1)
            x_group = x.reshape(x.size(0), self.group, -1)

            nfeat = F.normalize(x_group, p=2, dim=-1)
            ncenters = F.normalize(centroids_group, p=2, dim=-1)
            ncenters_sign = ncenters.sign()

            nfeat = nfeat.reshape(x.size(0), -1)
            ncenters = ncenters.reshape(self.nclass, -1)
            ncenters_sign = ncenters_sign.reshape(self.nclass, -1)

            logits_1 = torch.matmul(nfeat, ncenters.t()) / self.group
            logits_2 = torch.matmul(nfeat, ncenters_sign.t()) / self.group
            logits = (logits_1 + logits_2) * 0.5
        else:
            if self.input_group != 1:
                x_group = x.reshape(x.size(0), self.input_group, -1)
                nfeat = F.normalize(x_group, p=2, dim=-1)
                ncenters = F.normalize(centroids, p=2, dim=-1)

                nfeat = F.normalize(nfeat.reshape(x.size(0), -1), p=2, dim=-1)
                ncenters = ncenters.reshape(self.nclass, -1)
            else:
                centroids_group = centroids.reshape(self.nclass, self.group, -1)
                x_group = x.reshape(x.size(0), self.group, -1)

                nfeat = F.normalize(x_group, p=2, dim=-1)
                ncenters = F.normalize(centroids_group, p=2, dim=-1)

                nfeat = nfeat.reshape(x.size(0), -1)
                ncenters = ncenters.reshape(self.nclass, -1)

            logits = torch.matmul(nfeat, ncenters.t()) / self.group

        return logits

    def extra_repr(self) -> str:
        return 'in_features={}, n_class={}, learn_cent={}'.format(
            self.nfeat, self.nclass, self.learn_cent
        )
