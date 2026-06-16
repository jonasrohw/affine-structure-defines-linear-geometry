"""
Logit lens: apply ln_f + lm_head to intermediate residual streams.

Tests whether the MLP is needed by checking object prediction accuracy
at the post-attention residual (h₀ + attn_out, before MLP).

Usage (from salzburg/):
    python experiments/logit_lens.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import cfg as default_cfg
from src.data_generation import make_dataloader, token_type_mask
from src.model import build_model
from src.training import load_checkpoint


@torch.no_grad()
def evaluate_at_stages(model, loader, device, cfg, max_batches=50):
    """Evaluate object prediction accuracy at multiple residual stream stages.

    Stages:
      - h₀: post-embedding (layer 0)
      - h₀ + attn: post-attention, pre-MLP
      - h₀ + attn + mlp: full L1 residual (= normal model)
    """
    model.eval()
    ln_f = model.ln_f
    lm_head = model.lm_head

    stages = ["h0", "h0+attn", "h0+attn+mlp"]
    correct = {s: 0 for s in stages}
    total = 0

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        B, T = x.shape

        positions = torch.arange(T, device=device).unsqueeze(0)
        h0 = model.token_emb(x) + model.pos_emb(positions)

        block = model.blocks[0]
        attn_out = block.attn(block.ln1(h0))
        h0_plus_attn = h0 + attn_out
        mlp_out = block.mlp(block.ln2(h0_plus_attn))
        h1 = h0_plus_attn + mlp_out

        _, _, _, obj_mask = token_type_mask(y, cfg.vocab)
        total += obj_mask.sum().item()

        for stage_name, h in [("h0", h0), ("h0+attn", h0_plus_attn), ("h0+attn+mlp", h1)]:
            logits = lm_head(ln_f(h))
            preds = logits.argmax(dim=-1)
            correct[stage_name] += (preds[obj_mask] == y[obj_mask]).sum().item()

    return {s: correct[s] / max(1, total) for s in stages}, total


def main():
    cfg = default_cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg.train.checkpoint_dir / "best.pt"

    model = build_model(cfg.model, cfg.vocab)
    load_checkpoint(ckpt_path, model, device)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}\n")

    loader = make_dataloader(cfg, split="train", seed_offset=0)

    accs, total = evaluate_at_stages(model, loader, device, cfg, max_batches=50)

    print(f"Object prediction accuracy at different stages (n={total} object positions):\n")
    print(f"  h₀ (embedding only):      {accs['h0']:.4f}")
    print(f"  h₀ + attn (pre-MLP):      {accs['h0+attn']:.4f}")
    print(f"  h₀ + attn + mlp (full):   {accs['h0+attn+mlp']:.4f}")

    results_lines = [
        "# Logit Lens: Object Prediction Accuracy at Intermediate Stages",
        "",
        "Apply ln_f + lm_head to intermediate residual streams and check",
        "how often the argmax is the correct object token.",
        "",
        "Sequence format: `[x, r_k, a_i, b_j]` — prediction target is b_j at position 3.",
        "",
        f"Evaluated on {total} object positions.",
        "",
        "| Stage | Description | Object Accuracy |",
        "|-------|-------------|-----------------|",
        f"| h₀ | Embedding only (no transformer) | {accs['h0']:.4f} |",
        f"| h₀ + attn | Post-attention, pre-MLP | {accs['h0+attn']:.4f} |",
        f"| h₀ + attn + mlp | Full L1 residual (normal model) | {accs['h0+attn+mlp']:.4f} |",
        "",
        "## Interpretation",
        "",
    ]

    if accs["h0+attn"] > 0.95:
        results_lines.append(
            f"Post-attention accuracy is {accs['h0+attn']:.1%}, confirming that "
            "attention alone (plus ln_f + unembedding) is sufficient for object prediction. "
            "The MLP's contribution is marginal."
        )
    else:
        gap = accs["h0+attn+mlp"] - accs["h0+attn"]
        results_lines.append(
            f"Post-attention accuracy is {accs['h0+attn']:.1%}, "
            f"while the full model reaches {accs['h0+attn+mlp']:.1%} "
            f"(gap = {gap:.1%}). The MLP provides a non-trivial contribution."
        )

    cfg.train.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = cfg.train.results_dir / "logit_lens.md"
    with open(results_path, "w") as f:
        f.write("\n".join(results_lines) + "\n")
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
