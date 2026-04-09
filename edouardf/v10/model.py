"""
v6 model — 110M + ReLU² + U-Net skips + zero-init + value embeddings + x0 residual + per-layer lambdas

CONTRACT
--------------------
Must expose exactly one function:

    get_model(config: dict) -> torch.nn.Module

The returned model's forward method must have the signature:

    forward(idx: LongTensor[B, T], targets: LongTensor[B, T] | None = None)
        -> (logits: Tensor[B, T, vocab_size], loss: Tensor | None)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(seq_len, head_dim, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    hd = x.shape[-1]
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, seq_len, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.c_attn  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj  = nn.Linear(n_embd, n_embd,     bias=False)
        self.dropout = dropout
        cos, sin = precompute_rope(seq_len, self.head_dim)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
        self.val_embed = nn.Parameter(torch.zeros(seq_len, n_embd))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        v = v + self.val_embed[:T]
        hd = self.head_dim
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SquaredReLU(nn.Module):
    """ReLU²: relu(x)² — simpler than SwiGLU (2 matmuls vs 3), same param count at 4x width."""
    def __init__(self, n_embd, dropout):
        super().__init__()
        hidden = 4 * n_embd
        self.w_up   = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.w_up(x))
        return self.drop(self.w_down(h * h))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, seq_len, dropout):
        super().__init__()
        self.ln1  = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, seq_len, dropout)
        self.ln2  = RMSNorm(n_embd)
        self.mlp  = SquaredReLU(n_embd, dropout)
        self.lambda_attn = nn.Parameter(torch.ones(n_embd))
        self.lambda_mlp  = nn.Parameter(torch.ones(n_embd))
        self.lambda_x0 = nn.Parameter(torch.zeros(1))

    def forward(self, x, x0):
        x = x + self.lambda_attn * self.attn(self.ln1(x))
        x = x + self.lambda_mlp  * self.mlp(self.ln2(x))
        x = x + self.lambda_x0   * x0
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, n_layer, n_head, n_embd, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.n_layer = n_layer
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(vocab_size, n_embd),
            h    = nn.ModuleList([Block(n_embd, n_head, seq_len, dropout)
                                  for _ in range(n_layer)]),
            ln_f = RMSNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # U-Net skip connections: layer i <-> layer (n-1-i)
        # Learned scalars (init=0) so they activate only if useful
        half = n_layer // 2
        self.skip_weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(half)]
        )

        self.apply(self._init_weights)
        # Zero-init output projections (muP-like): model starts as near-identity
        for block in self.transformer.h:
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.zeros_(block.mlp.w_down.weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        n = self.n_layer
        half = n // 2
        saved = {}

        for i, block in enumerate(self.transformer.h):
            # U-Net: inject skip from symmetric first-half layer
            mirror = n - 1 - i
            if i >= half and mirror in saved:
                x = x + self.skip_weights[mirror] * saved[mirror]
            x = block(x, x0)
            if i < half:
                saved[i] = x

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        logits = 30.0 * torch.tanh(logits / 30.0)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def get_model(config: dict) -> nn.Module:
    return GPT(
        vocab_size = config.get("vocab_size", 32768),
        seq_len    = config.get("seq_len",    1024),
        n_layer    = config.get("n_layer",    12),
        n_head     = config.get("n_head",     12),
        n_embd     = config.get("n_embd",     768),
        dropout    = config.get("dropout",    0.0),
    )
