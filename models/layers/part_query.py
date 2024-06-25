import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMapPooling(nn.Module):
    def __init__(self, dim=-1, avgpool_size=0, out_type='max', topk_for_max=5):
        super().__init__()

        self.dim = dim
        if avgpool_size == 0:
            self.proj = nn.Identity()  # whether to perform avgpool
        else:
            self.proj = nn.AvgPool2d(avgpool_size, stride=1, padding=avgpool_size // 2)
        self.out_type = out_type
        self.topk_for_max = topk_for_max

    def forward(self, attn, value=None):
        """
        attn: (B, nh, Q, K)
        value: (B, nh, K, dim)
        return attn_pool: (B, nh, Q), attn_value: (B, nh, Q, dim)
        """
        B, nh, Q, K = attn.size()

        K_size = int(K ** 0.5)

        attn = attn.reshape(B, nh * Q, K_size, K_size)
        attn = self.proj(attn)
        attn = attn.reshape(B, nh, Q, -1)
        attn_value = None

        if self.out_type == 'max':
            attn_pool, max_idx = attn.max(dim=self.dim)
            if value is not None:
                # use softmax infinity?
                # attn_max_mask = (attn == attn_pool.unsqueeze(self.dim)).float()
                # attn_value = attn_max_mask @ value

                attn_value = value[torch.arange(value.shape[0]).reshape(-1, 1, 1),
                torch.arange(value.shape[1]).reshape(1, -1, 1),
                max_idx]  # (B, nh, Q, dim)
        elif self.out_type == 'topk_randmax':
            attn_pool = attn.topk(self.topk_for_max, dim=self.dim)[0]
            rand_mask = torch.rand_like(attn_pool)
            attn_pool = attn_pool * rand_mask
            attn_pool = attn_pool.max(dim=self.dim)[0]
        elif self.out_type == 'focal':
            attn_pool_max = attn.max(dim=self.dim)[0]
            attn_pool_mean = attn.mean(dim=self.dim)
            attn_pool = attn_pool_max - attn_pool_mean
        elif self.out_type == 'mean':
            attn_pool = attn.mean(dim=self.dim)
        else:
            raise NotImplementedError

        if attn_value is not None:
            return attn_pool, attn_value

        return attn_pool


class SinusoidalPositionalEncoding(nn.Module):
    """
    from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, embed_dim, max_len=196):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return x

    def extra_repr(self) -> str:
        return f'embed_dim={self.pe.size(2)}, max_len={self.pe.size(1)}'


class PartQuery(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 query_size: int,
                 num_heads: int = 1,
                 avgpool_size: int = 0,
                 out_type: str = 'max',
                 pe: bool = True,
                 track_stats: bool = False,
                 momentum: float = 0.01,
                 qv_linear: bool = False,
                 in_norm: bool = False,
                 lf_norm: bool = False,
                 learnable_scale: bool = False,
                 use_cossim: bool = False,
                 use_attn_norm: bool = False,
                 use_softmax: bool = False,
                 use_context_as_query: bool = False,
                 use_value: bool = False,
                 encoder_layers: int = 0,
                 softmax_scale: int = 0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.avgpool_size = avgpool_size
        self.out_type = out_type
        self.query_size = query_size
        self.pe = pe
        self.track_stats = track_stats
        self.momentum = momentum
        self.use_cossim = use_cossim
        self.use_attn_norm = use_attn_norm
        self.use_softmax = use_softmax
        self.use_context_as_query = use_context_as_query
        self.use_value = use_value
        self.encoder_layers = encoder_layers

        if self.encoder_layers > 0:
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(in_dim,
                                                                            nhead=8,
                                                                            dim_feedforward=in_dim,
                                                                            dropout=0.1,
                                                                            batch_first=True),
                                                 self.encoder_layers)
        else:
            self.encoder = nn.Identity()

        if qv_linear:
            if use_context_as_query:
                self.query = nn.Parameter(torch.randn(1, query_size, out_dim))
                self.query_linear = nn.Linear(out_dim, out_dim, bias=False)
                if self.use_value:
                    self.value_linear = nn.Linear(in_dim, out_dim, bias=False)
                else:
                    self.value_linear = nn.Linear(out_dim, out_dim, bias=False)
            else:
                self.query = nn.Parameter(torch.randn(1, query_size, in_dim))
                self.query_linear = nn.Linear(in_dim, in_dim, bias=False)
                self.value_linear = nn.Linear(in_dim, out_dim, bias=False)
        else:
            if use_value:
                logging.warning('use_value=True but has no effect')
            self.query = nn.Parameter(torch.randn(1, query_size, in_dim))
            self.value = nn.Parameter(torch.randn(1, query_size, out_dim))

        if in_norm:
            self.k_norm = nn.LayerNorm(in_dim)
        else:
            self.k_norm = nn.Identity()

        if self.use_context_as_query:
            self.k_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.k_proj = nn.Linear(in_dim, in_dim, bias=False)

        if lf_norm:
            self.lf_norm = nn.LayerNorm(out_dim)
        else:
            self.lf_norm = nn.Identity()

        if learnable_scale:
            self.scale = nn.Parameter(torch.ones([]))
        else:
            if self.use_softmax:
                if softmax_scale == 0:
                    self.scale = (self.out_dim // self.num_heads) ** -0.5
                else:
                    self.scale = softmax_scale
            else:
                self.scale = 1
        self.qv_linear = qv_linear

        if pe:
            self.pemb = SinusoidalPositionalEncoding(in_dim)
        else:
            self.pemb = nn.Identity()

        self.attn_pool = AttentionMapPooling(dim=-1,
                                             avgpool_size=avgpool_size,
                                             out_type=out_type)
        if self.use_attn_norm:
            self.attn_norm = nn.LayerNorm(query_size)
        else:
            self.attn_norm = nn.Identity()

        if self.use_softmax:
            self.attn_softmax = nn.Softmax(dim=-2)
        else:
            self.attn_softmax = nn.Identity()

        if self.track_stats:
            self.register_buffer('running_mean', torch.zeros(1, query_size, out_dim))
            self.register_buffer('running_var', torch.ones(1, query_size, out_dim))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, origx, return_attn=False, return_attn_pool=False, return_q_proj=False, query=None):
        """
        k: (B, K, d), K is number of tokens.
        """
        x = self.pemb(origx)
        x = self.encoder(x)

        if self.qv_linear:
            if self.use_value:
                if query is not None:
                    q = self.query_linear(query)
                    v = self.value_linear(self.k_norm(x))
                else:
                    q = self.query_linear(self.query)
                    v = self.value_linear(self.k_norm(x))
            else:
                if query is not None:
                    q = self.query_linear(query)
                    v = self.value_linear(query)
                else:
                    q = self.query_linear(self.query)
                    v = self.value_linear(self.query)
        else:
            q = self.query
            v = self.value

        BQ, Q, _ = q.size()
        B, K, _ = x.size()

        in_dim = self.in_dim
        num_heads = self.num_heads
        out_dim = self.out_dim

        q_proj = q.reshape(BQ, Q, num_heads, -1).transpose(1, 2)
        k_proj = self.k_proj(self.k_norm(x)).reshape(B, K, num_heads, -1).transpose(1, 2)
        if self.use_value:
            v_proj = v.reshape(B, K, num_heads, -1).transpose(1, 2)
        else:
            v_proj = v.reshape(BQ, Q, num_heads, -1).transpose(1, 2)

        if self.use_cossim:
            q_proj = F.normalize(q_proj, p=2, dim=-1)
            k_proj = F.normalize(k_proj, p=2, dim=-1)

        # q: (B, nh, Q, d // nh)
        # k: (B, nh, K, d // nh)
        attn = (q_proj @ k_proj.transpose(-2, -1)) * self.scale  # (B, nh, Q, K)
        attn = self.attn_norm(attn.transpose(2, 3).reshape(B * num_heads * K, -1)).reshape(B,
                                                                                           num_heads,
                                                                                           K,
                                                                                           Q).transpose(2, 3)

        if self.use_softmax:
            attn_pool = self.attn_pool(attn)
            attn = self.attn_softmax(attn)  # (B, nh, Q, K)
            local_feat = attn @ v_proj  # (B, nh, Q, dim)
        else:
            if self.use_value:
                attn_pool, attn_value = self.attn_pool(attn, v_proj)  # (B, nh, Q), (B, nh, Q, dim)
                local_feat = attn_pool.unsqueeze(3) * attn_value  # (B, nh, Q, out // nh)
            else:
                attn_pool = self.attn_pool(attn)  # (B, nh, Q)
                local_feat = attn_pool.unsqueeze(3) * v_proj  # (B, nh, Q, out // nh)

        local_feat = local_feat.transpose(1, 2)  # (B, Q, nh, out // nh)
        local_feat = local_feat.reshape(B, Q, -1)  # (B, Q, out)
        local_feat = self.lf_norm(local_feat)

        if self.training and self.track_stats:
            local_feat_var, local_feat_mean = torch.var_mean(local_feat, dim=0, unbiased=False, keepdim=True)

            difference = self.running_mean.data - local_feat_mean.data
            difference.mul_(self.momentum)
            self.running_mean.sub_(difference)

            difference = self.running_var.data - local_feat_var.data
            difference.mul_(self.momentum)
            self.running_var.sub_(difference)

        outputs = (local_feat,)

        if return_attn:
            outputs = outputs + (attn,)

        if return_attn_pool:
            outputs = outputs + (attn_pool,)

        if return_q_proj:
            outputs = outputs + (q_proj,)

        if len(outputs) == 1:
            return outputs[0]

        return outputs

    def sample_from_stats(self, bs, device=None):
        if self.track_stats:
            sample = torch.randn(bs, self.query_size, self.out_dim, device=device)
            reparam_sample = sample * (self.running_var + 1e-7).sqrt() + self.running_mean
            return reparam_sample
        else:
            c = torch.zeros(bs, self.query_size, self.out_dim, device=device)
            return c

    def extra_repr(self) -> str:
        return 'in_dim={}, out_dim={}, query_size={}, ' \
               'num_heads={}, avgpool_size={}, out_type={}, pe={}, ' \
               'track_stats={}, momentum={}, qv_linear={}, in_norm={}, lf_norm={}, ' \
               'learnable_scale={}, use_softmax={}, softmax_scale={}'.format(
            self.in_dim,
            self.out_dim,
            self.query_size,
            self.num_heads,
            self.avgpool_size,
            self.out_type,
            self.pe,
            self.track_stats,
            self.momentum,
            self.qv_linear,
            isinstance(self.k_norm, nn.LayerNorm),
            isinstance(self.lf_norm, nn.LayerNorm),
            isinstance(self.scale, nn.Parameter),
            self.use_softmax,
            self.scale
        )


class PartQueryV2(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 query_size: int,
                 num_heads: int = 1,
                 avgpool_size: int = 0,
                 out_type: str = 'max',
                 pe: bool = True,
                 track_stats: bool = False,
                 momentum: float = 0.01,
                 qv_linear: bool = False,
                 in_norm: bool = False,
                 lf_norm: bool = False,
                 learnable_scale: bool = False,
                 use_cossim: bool = False,
                 use_attn_norm: bool = False,
                 use_softmax: bool = False,
                 use_context_as_query: bool = False,
                 use_value: bool = False,
                 encoder_layers: int = 0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.avgpool_size = avgpool_size
        self.out_type = out_type
        self.query_size = query_size
        self.pe = pe
        self.track_stats = track_stats
        self.momentum = momentum
        self.use_cossim = use_cossim
        self.use_attn_norm = use_attn_norm
        self.use_softmax = use_softmax
        self.use_context_as_query = use_context_as_query
        self.use_value = use_value
        self.qv_linear = qv_linear
        self.encoder_layers = encoder_layers

        if self.encoder_layers > 0:
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(in_dim,
                                                                            nhead=8,
                                                                            dim_feedforward=in_dim,
                                                                            dropout=0.1,
                                                                            batch_first=True),
                                                 self.encoder_layers)
        else:
            self.encoder = nn.Identity()

        if qv_linear:
            if use_context_as_query:
                self.query = nn.Parameter(torch.randn(1, query_size, out_dim))
                self.query_linear = nn.Linear(out_dim, out_dim, bias=False)
                if self.use_value:
                    self.value_linear = nn.Linear(in_dim, out_dim, bias=False)
                else:
                    self.value_linear = nn.Linear(out_dim, out_dim, bias=False)
            else:
                self.query = nn.Parameter(torch.randn(1, query_size, in_dim))
                self.query_linear = nn.Linear(in_dim, in_dim, bias=False)
                self.value_linear = nn.Linear(in_dim, out_dim, bias=False)
        else:
            if use_value:
                logging.warning('use_value=True but has no effect')
            self.query = nn.Parameter(torch.randn(1, query_size, in_dim))
            self.value = nn.Parameter(torch.randn(1, query_size, out_dim))

        if in_norm:
            self.q_norm = nn.LayerNorm(in_dim)
            self.k_norm = nn.LayerNorm(in_dim)
            self.v_norm = nn.LayerNorm(in_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.v_norm = nn.Identity()

        if self.use_context_as_query:
            self.k_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.k_proj = nn.Linear(in_dim, in_dim, bias=False)

        if lf_norm:
            self.lf_norm = nn.LayerNorm(out_dim)
        else:
            self.lf_norm = nn.Identity()

        if learnable_scale:
            self.scale = nn.Parameter(torch.ones([]))
        else:
            if self.use_softmax:
                self.scale = (self.out_dim // self.num_heads) ** -0.5
            else:
                self.scale = 1

        if pe:
            self.pemb = SinusoidalPositionalEncoding(in_dim)
        else:
            self.pemb = nn.Identity()

        self.attn_pool = AttentionMapPooling(dim=-1,
                                             avgpool_size=avgpool_size,
                                             out_type=out_type)

        if self.use_softmax:
            self.attn_softmax = nn.Softmax(dim=-2)
        else:
            self.attn_softmax = nn.Identity()

    def forward(self, origx, return_attn=False, return_attn_pool=False, return_q_proj=False, query=None):
        """
        k: (B, K, d), K is number of tokens.
        """
        x = self.pemb(origx)
        x = self.encoder(x)

        if query is None:
            query = self.query

        if self.qv_linear:
            if self.use_value:
                q = self.query_linear(self.q_norm(query))
                v = self.value_linear(self.v_norm(x))
            else:
                q = self.query_linear(self.q_norm(query))
                v = self.value_linear(self.v_norm(query))
        else:
            q = query
            v = self.value

        BQ, Q, _ = q.size()
        B, K, _ = x.size()

        num_heads = self.num_heads

        q_proj = q.reshape(BQ, Q, num_heads, -1).transpose(1, 2)
        k_proj = self.k_proj(self.k_norm(x)).reshape(B, K, num_heads, -1).transpose(1, 2)
        if self.use_value:
            v_proj = v.reshape(B, K, num_heads, -1).transpose(1, 2)
        else:
            v_proj = v.reshape(BQ, Q, num_heads, -1).transpose(1, 2)

        if self.use_cossim:
            q_proj = F.normalize(q_proj, p=2, dim=-1)
            k_proj = F.normalize(k_proj, p=2, dim=-1)

        # q: (B, nh, Q, d // nh)
        # k: (B, nh, K, d // nh)
        attn = (q_proj @ k_proj.transpose(-2, -1)) * self.scale  # (B, nh, Q, K)

        if self.use_softmax:
            attn_pool = self.attn_pool(attn)
            attn = self.attn_softmax(attn)  # (B, nh, Q, K)
            local_feat = attn @ v_proj  # (B, nh, Q, dim)
        else:
            if self.use_value:
                attn_pool, attn_value = self.attn_pool(attn, v_proj)  # (B, nh, Q), (B, nh, Q, dim)
                local_feat = attn_pool.unsqueeze(3) * attn_value  # (B, nh, Q, out // nh)
            else:
                attn_pool = self.attn_pool(attn)  # (B, nh, Q)
                local_feat = attn_pool.unsqueeze(3) * v_proj  # (B, nh, Q, out // nh)

        local_feat = local_feat.transpose(1, 2)  # (B, Q, nh, out // nh)
        local_feat = local_feat.reshape(B, Q, -1)  # (B, Q, out)
        local_feat = self.lf_norm(local_feat)

        outputs = (local_feat,)

        if return_attn:
            outputs = outputs + (attn,)

        if return_attn_pool:
            outputs = outputs + (attn_pool,)

        if return_q_proj:
            outputs = outputs + (q_proj,)

        if len(outputs) == 1:
            return outputs[0]

        return outputs

    def extra_repr(self) -> str:
        return 'in_dim={}, out_dim={}, query_size={}, ' \
               'num_heads={}, avgpool_size={}, out_type={}, pe={}, ' \
               'momentum={}, qv_linear={}, in_norm={}, lf_norm={}, ' \
               'learnable_scale={}, use_softmax={}'.format(
            self.in_dim,
            self.out_dim,
            self.query_size,
            self.num_heads,
            self.avgpool_size,
            self.out_type,
            self.pe,
            self.momentum,
            self.qv_linear,
            isinstance(self.k_norm, nn.LayerNorm),
            isinstance(self.lf_norm, nn.LayerNorm),
            isinstance(self.scale, nn.Parameter),
            self.use_softmax
        )


if __name__ == '__main__':
    print(PartQuery(2048, 128, 64, 8)(torch.randn(1, 49, 2048)).size())
