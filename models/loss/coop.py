import torch
import torch.nn as nn
import torch.nn.functional as F


class LGHLoss(nn.Module):
    def __init__(self,
                 scale=1,
                 margin=0,
                 loss_scales=None,
                 avg_before_softmax=False,
                 lmbd=0.5,
                 ncontext=8,
                 exponential_scale=0,
                 div_method=0,
                 concept_cossim=True,
                 div_min=0,
                 avg_attn=False,
                 nregs=0,
                 **kwargs):
        super(LGHLoss, self).__init__()

        if loss_scales is None:
            loss_scales = {'logits': 1,
                           'hash_logits': 1,
                           'bin_logits': 1,
                           'cont_logits': 1,
                           'concept_logits': 0,
                           'attn_div_loss': 0}

        self.scale = scale
        self.margin = margin
        self.lmbd = lmbd
        self.loss_scales = loss_scales
        self.avg_before_softmax = avg_before_softmax
        self.ncontext = ncontext
        self.exponential_scale = exponential_scale
        self.div_method = div_method
        self.concept_cossim = concept_cossim
        self.div_min = div_min
        self.avg_attn = avg_attn
        self.nregs = nregs

        self.losses = {}

    def compute_margin_logits(self, logits, labels):
        if len(logits.size()) == 3:
            if len(labels.size()) == 2:
                y_onehot = labels.unsqueeze(0) * self.margin
                logits = self.scale * (logits - y_onehot)
                labels = labels / labels.sum(dim=-1, keepdim=True)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(-1, torch.unsqueeze(labels, dim=-1).unsqueeze(0), self.margin)
                logits = self.scale * (logits - y_onehot)
        else:
            if len(labels.size()) == 2:
                y_onehot = labels * self.margin
                logits = self.scale * (logits - y_onehot)
                labels = labels / labels.sum(dim=-1, keepdim=True)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.margin)
                logits = self.scale * (logits - y_onehot)

        return logits, labels

    def compute_ce_loss(self, logits, labels, cossim=True):
        if cossim:
            logits, labels = self.compute_margin_logits(logits, labels)
        else:
            if len(labels.size()) == 2:  # update label only
                labels = labels / labels.sum(dim=-1, keepdim=True)

        if len(logits.size()) == 3:
            loss = F.cross_entropy(logits.reshape(logits.size(0) * logits.size(1), -1),
                                   labels.unsqueeze(0).repeat(logits.size(0), 1).reshape(-1), reduction='none')

            if self.exponential_scale > 0:
                loss = loss.reshape(self.ncontext, -1).mean(dim=1)
                scale = torch.exp(-torch.arange(self.ncontext).flip(0) / self.exponential_scale).to(loss.device)
                loss = (scale * loss).sum()
            else:
                loss = loss.mean()
        else:
            loss = F.cross_entropy(logits, labels)
        return loss

    def compute_hash_loss(self, logits_1, logits_2, labels):
        if self.avg_before_softmax:
            logits = self.lmbd * logits_1 + (1 - self.lmbd) * logits_2
            loss = self.compute_ce_loss(logits, labels)
        else:
            logits_1, _ = self.compute_margin_logits(logits_1, labels)
            logits_2, labels = self.compute_margin_logits(logits_2, labels)

            if len(labels.size()) == 1:
                labels = F.one_hot(labels, num_classes=logits_1.shape[-1])

            prob_1 = torch.softmax(logits_1, dim=-1)
            prob_2 = torch.softmax(logits_2, dim=-1)
            prob = self.lmbd * prob_1 + (1 - self.lmbd) * prob_2
            log_prob = torch.log(prob.clamp(min=1e-7))  # clamp for numerical stability
            if len(logits_1.size()) == 3:
                if self.exponential_scale > 0:
                    loss = - labels.unsqueeze(0) * log_prob  # (Q, N, C)
                    loss = loss.sum(dim=-1).mean(dim=1)  # (Q,)

                    scale = torch.exp(-torch.arange(loss.size(0)).flip(0) / self.exponential_scale).to(loss.device)
                    loss = (scale * loss).sum()
                else:
                    loss = - labels.unsqueeze(0) * log_prob  # (Q, N, C)
                    loss = loss.sum(dim=-1).mean()
            else:
                loss = - labels * log_prob  # (N, C)
                loss = loss.sum(dim=-1).mean()

        return loss

    def forward(self, outputs, labels):
        with torch.no_grad():
            quan_sim = 1 - F.cosine_similarity(outputs['codes'], outputs['codes'].sign(), dim=-1).mean()
            self.losses['quan'] = quan_sim

        device = outputs['codes'].device
        total_loss = torch.tensor(0.).to(device)

        if 'logits' in self.loss_scales and self.loss_scales['logits'] != 0:
            aux_loss = self.compute_ce_loss(outputs['logits'], labels)
            total_loss += aux_loss * self.loss_scales['logits']
            self.losses['aux'] = aux_loss

        if 'concept_logits' in self.loss_scales and self.loss_scales['concept_logits'] != 0:
            concept_loss = self.compute_ce_loss(outputs['logits_concept'], labels, cossim=self.concept_cossim)
            total_loss += concept_loss * self.loss_scales['concept_logits']
            self.losses['concept'] = concept_loss

        if 'filip_logits' in self.loss_scales and self.loss_scales['filip_logits'] != 0:
            # filip_loss = self.compute_ce_loss(outputs['logits_filip'], labels, cossim=True)

            filip_loss_i2t = self.compute_ce_loss(outputs['logits_filip_i2t'], labels, cossim=True)
            filip_loss_t2i = self.compute_ce_loss(outputs['logits_filip_t2i'], labels, cossim=True)
            filip_loss = (filip_loss_t2i + filip_loss_i2t) * 0.5

            total_loss += filip_loss * self.loss_scales['filip_logits']

            self.losses['filip'] = filip_loss

        if 'hash_logits' in self.loss_scales and self.loss_scales['hash_logits'] != 0:
            hash_loss = self.compute_hash_loss(outputs['logits_cont'], outputs['logits_bin'], labels)
            total_loss += hash_loss * self.loss_scales['hash_logits']
            self.losses['hash'] = hash_loss

        if 'cont_logits' in self.loss_scales and self.loss_scales['cont_logits'] != 0:
            hash_loss = self.compute_ce_loss(outputs['logits_cont'], labels)
            total_loss += hash_loss * self.loss_scales['cont_logits']
            self.losses['cont'] = hash_loss

        if 'bin_logits' in self.loss_scales and self.loss_scales['bin_logits'] != 0:
            hash_loss = self.compute_ce_loss(outputs['logits_bin'], labels)
            total_loss += hash_loss * self.loss_scales['bin_logits']
            self.losses['bin'] = hash_loss

        if 'attn_div_loss' in self.loss_scales and self.loss_scales['attn_div_loss'] != 0:
            if self.avg_attn:
                last_attn = torch.stack(outputs['attn_cache'], dim=0).mean(dim=0)  # (B, 12, 58, 58)
            else:
                last_attn = outputs['attn_cache'][-1]  # (B, 12, 58, 58)

            if self.nregs != 0:
                last_attn = last_attn[:, :, -self.ncontext - self.nregs:-self.nregs,
                            1:-self.ncontext - self.nregs]  # (B, 12, Q, 49)
            else:
                last_attn = last_attn[:, :, -self.ncontext:, 1:-self.ncontext]  # (B, 12, Q, 49)

            avg_attn = last_attn.mean(dim=1)  # (B, Q, 49)
            avg_attn_l2 = F.normalize(avg_attn, dim=-1, p=2)
            cossim = avg_attn_l2 @ avg_attn_l2.transpose(1, 2)  # (B, Q, Q)
            if self.div_method == 0:
                cossim = (cossim - self.div_min).relu()  # .mean()
            cossim = cossim.mean(dim=0)  # (Q, Q)
            mask = torch.ones_like(cossim).bool()
            triu_mask = torch.triu(mask, 1)
            triu_cossim = cossim[triu_mask]
            div_loss = triu_cossim.mean()
            self.losses['attn_div'] = div_loss
            total_loss += div_loss * self.loss_scales['attn_div_loss']

        return total_loss


class LGHv3Loss(LGHLoss):
    def forward(self, outputs, true_labels):
        B = outputs['codes'].size(0)
        device = outputs['logits'].device
        labels = F.one_hot(torch.arange(B), B).to(device)  # is a contrastive loss, so only diagonal one hot

        return super().forward(outputs, labels)
