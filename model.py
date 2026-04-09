"""
Key changes from baseline:
- RMSNorm instead of LayerNorm 
- SwiGLU MLP instead of GELU
- QK-norm for training stability at high LR
- Vocab padded to multiple of 128 for GPU efficiency
- No dropout (wasteful at this scale/time budget)

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


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # QK-norm for stable training at high LR (from modded-nanogpt)
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP — same param count as 4x GELU MLP but better quality."""
    def __init__(self, n_embd):
        super().__init__()
        hidden = int(8 / 3 * n_embd)
        hidden = ((hidden + 255) // 256) * 256  # round up for GPU efficiency
        self.w1 = nn.Linear(n_embd, hidden, bias=False)  # gate
        self.w2 = nn.Linear(n_embd, hidden, bias=False)  # up
        self.w3 = nn.Linear(hidden, n_embd, bias=False)  # down

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.rms1 = nn.RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.rms2 = nn.RMSNorm(n_embd)
        self.mlp = SwiGLUMLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.rms1(x))
        x = x + self.mlp(self.rms2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, n_layer, n_head, n_embd):
        super().__init__()
        self.seq_len = seq_len
        self.padded_vocab = ((vocab_size + 127) // 128) * 128
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.padded_vocab, n_embd),
            wpe=nn.Embedding(seq_len, n_embd),
            h=nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)]),
        ))
        self.rms_f = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, self.padded_vocab, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layer)
        for block in self.transformer.h:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
            nn.init.normal_(block.mlp.w3.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.rms_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# Competition interface
# ---------------------------------------------------------------------------

def get_model(config: dict) -> nn.Module:
    return GPT(
        vocab_size=config.get("vocab_size", 32768),
        seq_len=config.get("seq_len", 1024),
        n_layer=config.get("n_layer", 12),
        n_head=config.get("n_head", 12),
        n_embd=config.get("n_embd", 768),
    )
