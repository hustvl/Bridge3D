import numpy as np
from torch import nn
from torch.nn import functional as F

from .multihead_custom_attention import MultiheadCustomAttention
from pdb import set_trace


class AdaLN(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.modulation = nn.Sequential(
             nn.SiLU(), nn.Linear(embedding_dim, 2 * embedding_dim, bias=True)
        )
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x, t):
        """
        Args:
            x: A tensor of shape (N, B, C)
            t: A tensor of shape (B, C)
        """
        scale, shift = self.modulation(t).chunk(2, dim=-1)  # (B, C), (B, C)
        x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        return x  

class FeedforwardLayer(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, dropout=0.0,
                 use_adaln=False):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.activation = F.relu
        self._reset_parameters()
        if use_adaln:
            self.adaln = AdaLN(embedding_dim)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, diff_ts=None):
        if diff_ts is not None:
            x = self.adaln(x, diff_ts)
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = x + self.dropout(output)
        output = self.norm(output)
        return output


class RelativeCrossAttentionLayer(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout=0.0, use_adaln=False):
        super().__init__()
        self.multihead_attn = MultiheadCustomAttention(
            embedding_dim, num_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        if use_adaln:
            self.adaln = AdaLN(embedding_dim)

    def forward(self, query, value, diff_ts=None,
                query_pos=None, value_pos=None, pad_mask=None, attn_bias=None):
        if diff_ts is not None:
            adaln_query = self.adaln(query, diff_ts)
        else:
            adaln_query = query
        # attn_output, _ = self.multihead_attn(
        #     query=adaln_query,
        #     key=value,
        #     value=value,
        #     rotary_pe=None if query_pos is None else (query_pos, value_pos),
        #     key_padding_mask=pad_mask,
        #     attn_bias=attn_bias
        # )
        
        attn_output, _ = self.multihead_attn(
            query=adaln_query,
            key=value,
            value=value,
            rotary_pe=None if value_pos is None else (query_pos, value_pos),
            key_padding_mask=pad_mask,
            attn_bias=attn_bias
        )

        output = query + self.dropout(attn_output)
        output = output.to(self.norm.weight.dtype)
        output = self.norm(output)
        return output


class FFWRelativeCrossAttentionModule(nn.Module):

    def __init__(self, embedding_dim, num_attn_heads, num_layers,
                 use_adaln=True):
        super().__init__()

        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(RelativeCrossAttentionLayer(
                embedding_dim, num_attn_heads, use_adaln=use_adaln
            ))
            self.ffw_layers.append(FeedforwardLayer(
                embedding_dim, embedding_dim, use_adaln=use_adaln
            ))

    def forward(self, query, value, diff_ts=None,
                query_pos=None, value_pos=None, pad_mask=None, attn_bias=None):
        output = []
        for i in range(self.num_layers):
            query = self.attn_layers[i](
                query, value, diff_ts, query_pos, value_pos, pad_mask=pad_mask, attn_bias=attn_bias
            )
            query = self.ffw_layers[i](query, diff_ts)
            output.append(query)
        return output
