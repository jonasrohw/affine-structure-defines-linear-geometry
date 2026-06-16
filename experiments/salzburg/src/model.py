"""
Minimal GPT-style causal transformer for the Vienna experiment.

The model exposes its residual stream at every layer via forward hooks,
which are used by representations.py to extract subject/object representations.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, VocabConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.hidden_dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.hidden_dim // cfg.n_heads

        self.qkv = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        # Reshape to (B, n_heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        att = att.masked_fill(~causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, n_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_dim, 4 * cfg.hidden_dim)
        self.fc2 = nn.Linear(4 * cfg.hidden_dim, cfg.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, capture_sublayers: Optional[dict] = None) -> torch.Tensor:
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
        if capture_sublayers is not None:
            capture_sublayers["attn"].append(attn_out.detach().clone())
            capture_sublayers["mlp"].append(mlp_out.detach().clone())
        return x


class MiniTransformer(nn.Module):
    """Small GPT-style decoder-only transformer.

    Attributes:
        residuals: When set (via capture_residuals context manager), filled with
                   per-layer residual stream tensors during forward pass.
                   residuals[0] = embedding output (before any block),
                   residuals[k] = output of block k-1 (after block k-1).
    """

    def __init__(self, model_cfg: ModelConfig, vocab_cfg: VocabConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.vocab_cfg = vocab_cfg

        self.token_emb = nn.Embedding(vocab_cfg.vocab_size, model_cfg.hidden_dim)
        self.pos_emb = nn.Embedding(model_cfg.context_len, model_cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(model_cfg) for _ in range(model_cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(model_cfg.hidden_dim)

        if model_cfg.tie_embeddings:
            self.lm_head = nn.Linear(model_cfg.hidden_dim, vocab_cfg.vocab_size, bias=False)
            self.lm_head.weight = self.token_emb.weight
        else:
            self.lm_head = nn.Linear(model_cfg.hidden_dim, vocab_cfg.vocab_size, bias=False)

        self._capture: Optional[List[torch.Tensor]] = None
        self._capture_sublayers: Optional[dict] = None
        self._apply_init()

    def _apply_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) long tensor of token ids.

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        assert T <= self.model_cfg.context_len, f"Sequence length {T} exceeds context_len {self.model_cfg.context_len}"

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Capture layer 0 = post-embedding residual
        if self._capture is not None:
            self._capture.append(x.detach().clone())

        for block in self.blocks:
            x = block(x, capture_sublayers=self._capture_sublayers)
            if self._capture is not None:
                self._capture.append(x.detach().clone())

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def capture_residuals(self):
        """Context manager: fills self._capture with residual tensors during forward.

        Usage:
            with model.capture_residuals() as residuals:
                logits = model(input_ids)
            # residuals[0] = post-embedding, residuals[k] = after block k-1
        """
        return _ResidualCapture(self)

    def capture_sublayers(self):
        """Context manager: captures per-block attention and MLP outputs.

        Usage:
            with model.capture_sublayers() as sublayers:
                logits = model(input_ids)
            # sublayers["attn"][k] = attention output of block k
            # sublayers["mlp"][k] = MLP output of block k
        """
        return _SublayerCapture(self)

    @property
    def n_layers(self):
        return self.model_cfg.n_layers

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


class _ResidualCapture:
    def __init__(self, model: MiniTransformer):
        self.model = model

    def __enter__(self):
        self.model._capture = []
        return self.model._capture

    def __exit__(self, *args):
        self.model._capture = None


class _SublayerCapture:
    def __init__(self, model: MiniTransformer):
        self.model = model

    def __enter__(self):
        self.model._capture_sublayers = {"attn": [], "mlp": []}
        return self.model._capture_sublayers

    def __exit__(self, *args):
        self.model._capture_sublayers = None


def build_model(model_cfg: ModelConfig, vocab_cfg: VocabConfig) -> MiniTransformer:
    return MiniTransformer(model_cfg, vocab_cfg)
