import torch
import torch.nn as nn


class ExchNetLocalExchange(nn.Module):
    def __init__(self, attention_size, channels, nclass, p=0.5, min_count=50):
        super().__init__()

        self.min_count = min_count
        self.register_buffer('run_count', torch.tensor(0))
        self.p = p

        anchor = torch.zeros(nclass, attention_size, channels)
        self.register_buffer('anchor', anchor)

        cache = torch.zeros(nclass, attention_size, channels)
        self.register_buffer('cache', cache)

        count = torch.zeros(nclass)  # count
        self.register_buffer('count', count)

    def update_anchor(self):
        self.anchor.data.copy_(self.cache / self.count.unsqueeze(1).unsqueeze(2))
        self.reset_stats()

    def reset_stats(self):
        self.cache.fill_(0)
        self.count.fill_(0)

    def forward(self, features, labels):
        # features = (b, m, c)
        # labels = (b, nc)

        if self.training:
            self.run_count += 1

        if self.training and self.run_count.item() >= self.min_count:
            with torch.no_grad():
                # (b, 1, m, c) * (b, nc, 1, 1) = (b, nc, m, c) -> (nc, m, c)
                batch_cache = (features.unsqueeze(1) * labels.unsqueeze(2).unsqueeze(3)).sum(dim=0)
                self.cache += batch_cache

                batch_count = labels.sum(dim=0)
                self.count += batch_count

            exchange_mask = torch.rand(features.size(0), features.size(1), device=features.device) > self.p
            exchange_mask = exchange_mask.float().unsqueeze(2)

            exch_features = exchange_mask * self.anchor[labels.argmax(1)] + (1 - exchange_mask) * features
            return exch_features
        else:  # no changes
            return features
