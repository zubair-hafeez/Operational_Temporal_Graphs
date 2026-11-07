"""Matched node, set, typed graph, and temporal-attention survival controls.

These are independent target adaptations, not reproductions of named published
TGN/TGAT/DyGFormer systems. Group incidence pooling is bipartite message passing.
"""
import numpy as np
import torch
from torch import nn


def relation_summary(x, relations):
    out = [x, np.broadcast_to(x.mean(0), x.shape)]
    for j in range(relations.shape[1]):
        r = relations[:, j].copy()
        r[r < 0] = np.arange((r < 0).sum()) + r.max() + 1
        _, codes = np.unique(r, return_inverse=True)
        counts = np.bincount(codes)
        total = np.zeros((len(counts), x.shape[1]))
        np.add.at(total, codes, x)
        out.extend([total[codes] / counts[codes, None], np.log1p(counts[codes, None])])
    return np.concatenate(out, axis=1).astype(np.float32)


def local_codes(relations):
    out = []
    for j in range(relations.shape[1]):
        r = relations[:, j].copy()
        missing = r < 0
        r[missing] = np.arange(missing.sum()) + r.max() + 1
        out.append(np.unique(r, return_inverse=True)[1])
    return np.column_stack(out)


def pool(x, r):
    n = int(r.max()) + 1
    sums = x.new_zeros(n, x.shape[-1]).index_add_(0, r, x)
    counts = torch.bincount(r, minlength=n).clamp_min(1)
    return sums[r] / counts[r, None]


class CohortNet(nn.Module):
    def __init__(self, features, relations, architecture, width=48):
        super().__init__()
        self.architecture, self.relations = architecture, relations
        self.embed = nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.LayerNorm(width))
        scopes = 1 if architecture == "mlp" else 2 + relations
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(width*scopes, width), nn.GELU(), nn.LayerNorm(width)) for _ in range(2)])
        if architecture == "temporal_attention":
            self.token = nn.Linear(4, width)
            self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
            self.fuse = nn.Linear(width*(1+relations), width)
        self.head = nn.Linear(width, 5)

    def forward(self, x, r, history):
        h = self.embed(x)
        if self.architecture == "temporal_attention":
            blocks = [h]
            for j in range(self.relations):
                token = self.token(history[:, j, :, :4])
                mask = history[:, j, :, 4] == 0
                empty = mask.all(-1)
                # Give empty histories one harmless padding token; zero output explicitly.
                mask = mask.clone(); mask[empty, -1] = False
                att, _ = self.attention(h[:, None], token, token, key_padding_mask=mask, need_weights=False)
                blocks.append(att[:, 0] * (~empty)[:, None])
            h = torch.nn.functional.gelu(self.fuse(torch.cat(blocks, -1)))
        for layer in self.layers:
            if self.architecture == "mlp":
                values = [h]
            elif self.architecture == "set":
                global_h = h.mean(0, keepdim=True).expand_as(h)
                values = [h, global_h] + [global_h for _ in range(self.relations)]
            else:
                values = [h, h.mean(0, keepdim=True).expand_as(h)] + [pool(h, r[:, j]) for j in range(self.relations)]
            h = h + layer(torch.cat(values, -1))
        return self.head(h)
