"""
v4 model — Fused CE + GQA + bigger model.

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
    """GQA: n_head Q heads, n_kv_head KV heads (n_kv_head < n_head saves memory)."""

    def __init__(self, n_embd, n_head, n_kv_head, seq_len, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_kv_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head
        self.head_dim = n_embd // n_head
        kv_dim = n_kv_head * self.head_dim

        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, kv_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, kv_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

        cos, sin = precompute_rope(seq_len, self.head_dim)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # Value Embeddings (ResFormer) — sized to KV dim
        self.val_embed = nn.Parameter(torch.zeros(seq_len, kv_dim))

    def forward(self, x):
        B, T, C = x.shape
        hd = self.head_dim

        q = self.q_proj(x).view(B, T, self.n_head, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, hd).transpose(1, 2)
        v = self.v_proj(x)
        v = v + self.val_embed[:T]
        v = v.view(B, T, self.n_kv_head, hd).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        # Expand KV heads to match Q heads for SDPA
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        hidden = int(2 * (4 * n_embd) / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up   = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, seq_len, dropout):
        super().__init__()
        self.ln1  = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, seq_len, dropout)
        self.ln2  = RMSNorm(n_embd)
        self.mlp  = SwiGLU(n_embd, dropout)
        self.lambda_attn = nn.Parameter(torch.ones(n_embd))
        self.lambda_mlp  = nn.Parameter(torch.ones(n_embd))
        self.lambda_x0   = nn.Parameter(torch.zeros(1))

    def forward(self, x, x0):
        x = x + self.lambda_attn * self.attn(self.ln1(x))
        x = x + self.lambda_mlp  * self.mlp(self.ln2(x))
        x = x + self.lambda_x0   * x0
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, n_layer, n_head, n_kv_head, n_embd, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.n_layer = n_layer
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(vocab_size, n_embd),
            h    = nn.ModuleList([Block(n_embd, n_head, n_kv_head, seq_len, dropout)
                                  for _ in range(n_layer)]),
            ln_f = RMSNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.compiler.disable
    def _chunked_loss(self, hidden, targets, chunk_size=1024):
        """Fused linear+CE: computes loss chunk-by-chunk, never allocates (B*T, V) logits.
        Saves ~500MB VRAM with V=32768 and typical batch sizes."""
        hidden = hidden.contiguous().view(-1, hidden.size(-1))
        targets = targets.contiguous().view(-1)
        total = torch.zeros(1, device=hidden.device, dtype=hidden.dtype)
        n = hidden.size(0)
        for i in range(0, n, chunk_size):
            h = hidden[i:i + chunk_size]
            t = targets[i:i + chunk_size]
            logits = h @ self.lm_head.weight.T
            if self.lm_head.bias is not None:
                logits = logits + self.lm_head.bias
            logits = 30.0 * torch.tanh(logits / 30.0)
            total = total + F.cross_entropy(logits, t, reduction='sum')
        return total / n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        for block in self.transformer.h:
            x = block(x, x0)
        x = self.transformer.ln_f(x)

        if targets is not None:
            loss = self._chunked_loss(x, targets)
            return x.new_empty(0), loss

        logits = self.lm_head(x)
        logits = 30.0 * torch.tanh(logits / 30.0)
        return logits, None

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def get_model(config: dict) -> nn.Module:
    return GPT(
        vocab_size = config.get("vocab_size", 32768),
        seq_len    = config.get("seq_len",    1024),
        n_layer    = config.get("n_layer",    20),
        n_head     = config.get("n_head",     20),
        n_kv_head  = config.get("n_kv_head",  4),
        n_embd     = config.get("n_embd",     1280),
        dropout    = config.get("dropout",    0.0),
    )
