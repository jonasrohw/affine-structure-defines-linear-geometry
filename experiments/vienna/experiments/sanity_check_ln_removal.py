"""
Sanity check: does removing ln_f at inference change object prediction accuracy?

Also measures the direct cosine between f2 residual vectors and lm_head rows
to verify the directional model of next-token prediction.

Usage (from vienna/):
    python experiments/sanity_check_ln_removal.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.config import cfg as default_cfg
from src.data_generation import TOKENS_PER_TRIPLE, generate_sequence
from src.model import build_model
from src.representations import get_filler_representations
from src.training import load_checkpoint


def eval_accuracy_on_sequences(model, cfg, device, use_ln_f=True):
    """Evaluate object prediction accuracy on generated sequences.

    Returns (total_correct, total_object_positions).
    """
    import random
    rng = random.Random(999)
    n_sequences = 500
    all_indices = list(range(cfg.vocab.n_subjects))

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for _ in range(n_sequences):
            tokens = generate_sequence(cfg.train.seq_length, all_indices, cfg.vocab, rng)
            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor(tokens[1:], dtype=torch.long, device=device)

            B, T = input_ids.shape
            positions = torch.arange(T, device=device).unsqueeze(0)
            x = model.token_emb(input_ids) + model.pos_emb(positions)
            for block in model.blocks:
                x = block(x)

            if use_ln_f:
                x = model.ln_f(x)

            logits = model.lm_head(x)
            preds = logits.argmax(dim=-1).squeeze(0)

            # Object positions: positions 6, 13, 20, ... (0-indexed in the input)
            # In the sequence, object is at index 6, 13, 20, ... within the full token list
            # In input (tokens[:-1]), object position 6 predicts tokens[7], but we want
            # the position that predicts the object. The object is at position 6 in the
            # triple (0-indexed), so in input, position 5 (f2) predicts the object at position 6.
            for triple_start in range(0, len(tokens) - 1, TOKENS_PER_TRIPLE):
                obj_pred_pos = triple_start + 5  # f2 position, predicts object
                if obj_pred_pos < T:
                    pred_token = preds[obj_pred_pos].item()
                    true_token = targets[obj_pred_pos].item()
                    if pred_token == true_token:
                        correct += 1
                    total += 1

    return correct, total


def main():
    cfg = default_cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg.train.checkpoint_dir / "best.pt"

    model = build_model(cfg.model, cfg.vocab)
    load_checkpoint(ckpt_path, model, device)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}\n")

    results_lines = []

    def log(line=""):
        print(line)
        results_lines.append(line)

    # ── Accuracy with vs without ln_f ─────────────────────────────────
    log("=" * 60)
    log("Object prediction accuracy: with vs without ln_f")
    log("=" * 60)

    correct_with, total_with = eval_accuracy_on_sequences(model, cfg, device, use_ln_f=True)
    correct_without, total_without = eval_accuracy_on_sequences(model, cfg, device, use_ln_f=False)

    acc_with = correct_with / total_with
    acc_without = correct_without / total_without

    log(f"With ln_f:    {correct_with}/{total_with} = {acc_with:.4f}")
    log(f"Without ln_f: {correct_without}/{total_without} = {acc_without:.4f}")
    log()

    # ── Direct cosine between f2 vectors and lm_head rows ────────────
    log("=" * 60)
    log("Direct cosine: f2 residual vs lm_head[b_j]")
    log("=" * 60)

    all_indices = list(range(cfg.vocab.n_subjects))
    filler_reprs = get_filler_representations(model, cfg, all_indices, device)
    lm_head_w = model.lm_head.weight.detach().cpu()  # (vocab_size, d)

    n_snapshots = len(filler_reprs[(0, 0)])
    layer_labels = ["emb"] + [f"L{i+1}" for i in range(n_snapshots - 1)]

    for layer in range(n_snapshots):
        cos_correct = []
        rank_correct = []

        for i in all_indices:
            for k in range(cfg.vocab.num_relations):
                j = cfg.vocab.apply_relation(k, i)
                f2_vec = filler_reprs[(i, k)][layer]
                obj_row = lm_head_w[cfg.vocab.object_id(j)]

                # Cosine with correct object's unembedding row
                cos = F.cosine_similarity(f2_vec.unsqueeze(0), obj_row.unsqueeze(0)).item()
                cos_correct.append(cos)

                # Rank: how many lm_head rows have higher dot product?
                dots = lm_head_w @ f2_vec
                rank = (dots > dots[cfg.vocab.object_id(j)]).sum().item() + 1
                rank_correct.append(rank)

        cos_arr = np.array(cos_correct)
        rank_arr = np.array(rank_correct)

        log(f"Layer {layer_labels[layer]}:")
        log(f"  cos(f2, lm_head[b_j]):  mean={cos_arr.mean():.4f}  min={cos_arr.min():.4f}  max={cos_arr.max():.4f}")
        log(f"  dot-product rank of b_j: mean={rank_arr.mean():.1f}  top-1 acc={( rank_arr == 1).mean():.4f}")

        # Same but with ln_f applied
        ln_f = model.ln_f.cpu()
        cos_correct_ln = []
        rank_correct_ln = []
        for i in all_indices:
            for k in range(cfg.vocab.num_relations):
                j = cfg.vocab.apply_relation(k, i)
                f2_vec = filler_reprs[(i, k)][layer]
                with torch.no_grad():
                    f2_ln = ln_f(f2_vec.unsqueeze(0)).squeeze(0)
                obj_row = lm_head_w[cfg.vocab.object_id(j)]

                cos = F.cosine_similarity(f2_ln.unsqueeze(0), obj_row.unsqueeze(0)).item()
                cos_correct_ln.append(cos)

                dots = lm_head_w @ f2_ln
                rank = (dots > dots[cfg.vocab.object_id(j)]).sum().item() + 1
                rank_correct_ln.append(rank)

        cos_ln = np.array(cos_correct_ln)
        rank_ln = np.array(rank_correct_ln)

        log(f"  (post-LN) cos(ln_f(f2), lm_head[b_j]):  mean={cos_ln.mean():.4f}  min={cos_ln.min():.4f}  max={cos_ln.max():.4f}")
        log(f"  (post-LN) dot-product rank of b_j: mean={rank_ln.mean():.1f}  top-1 acc={(rank_ln == 1).mean():.4f}")
        log()

    # Save results
    cfg.train.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = cfg.train.results_dir / "sanity_check_ln_removal.md"
    with open(results_path, "w") as f:
        f.write("# Sanity Check: ln_f Removal at Inference\n\n")
        f.write("```\n")
        f.write("\n".join(results_lines))
        f.write("\n```\n")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
