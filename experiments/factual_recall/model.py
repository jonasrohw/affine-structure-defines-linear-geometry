"""Small causal transformer used by the factual recall experiments."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TokenVocabulary, TransformerConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.hidden_dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(hidden_dim, dim=2)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        att = att.masked_fill(~causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_dim, 4 * cfg.hidden_dim)
        self.fc2 = nn.Linear(4 * cfg.hidden_dim, cfg.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalTransformer(nn.Module):
    """GPT-style decoder used by both reported experiments."""

    def __init__(self, model_cfg: TransformerConfig, vocab_cfg: TokenVocabulary) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.vocab_cfg = vocab_cfg
        self.token_emb = nn.Embedding(vocab_cfg.vocab_size, model_cfg.hidden_dim)
        self.pos_emb = nn.Embedding(model_cfg.context_len, model_cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(model_cfg) for _ in range(model_cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(model_cfg.hidden_dim)
        self.lm_head = nn.Linear(model_cfg.hidden_dim, vocab_cfg.vocab_size, bias=False)
        if model_cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight
        self._apply_init()

    def _apply_init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.model_cfg.context_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context_len {self.model_cfg.context_len}"
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    @property
    def n_layers(self) -> int:
        return self.model_cfg.n_layers

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(model_cfg: TransformerConfig, vocab_cfg: TokenVocabulary) -> CausalTransformer:
    return CausalTransformer(model_cfg, vocab_cfg)
