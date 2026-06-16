"""
Accuracy of affine-predicted representations when fed to lm_head.

Question: if we replace the model's internal computation with an affine
transformation from L0 (embedding) to the target representation, and then
apply lm_head, how often is the correct object predicted?

Usage (from salzburg/):
    python experiments/affine_accuracy.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.affine import fit_affine
from src.config import cfg as default_cfg
from src.model import build_model
from src.representations import (
    get_subject_representations,
    get_subject_sublayer_outputs,
    stack_subject_representations,
    stack_all_subject_representations,
)
from src.training import load_checkpoint

RELATION_KEYS = ["R0", "R1", "R2"]


def affine_accuracy_per_relation(S, target, lm_head, ln_f, vocab, relation_idx, subject_indices, device, k=5):
    """K-fold CV: fit affine S→target, apply lm_head, measure object accuracy."""
    n = S.shape[0]
    fold_size = n // k
    indices = np.arange(n)
    correct = 0
    total = 0

    for fold in range(k):
        val_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        W, b = fit_affine(S[train_idx], target[train_idx], bias=True)

        S_val = S[val_idx]
        O_hat = S_val @ W.T
        if b is not None:
            O_hat = O_hat + b

        with torch.no_grad():
            if ln_f is not None:
                O_hat = ln_f(O_hat.to(device)).cpu()
            logits = O_hat @ lm_head.T
            preds = logits.argmax(dim=1)

        for idx_in_fold, global_idx in enumerate(val_idx):
            subj_idx = subject_indices[global_idx]
            expected_obj = vocab.apply_relation(relation_idx, subj_idx)
            expected_token = vocab.object_id(expected_obj)
            if preds[idx_in_fold].item() == expected_token:
                correct += 1
            total += 1

    return correct / max(1, total)


def main():
    cfg = default_cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg.train.checkpoint_dir / "best.pt"

    model = build_model(cfg.model, cfg.vocab)
    load_checkpoint(ckpt_path, model, device)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}\n")

    all_indices = list(range(cfg.vocab.n_subjects))
    n_rel = cfg.vocab.num_relations

    lm_head_w = model.lm_head.weight.detach().cpu()  # (vocab_size, hidden_dim)
    ln_f = model.ln_f

    print("Extracting representations...")
    subj_reprs = get_subject_representations(model, cfg, all_indices, device)
    subj_sublayers = get_subject_sublayer_outputs(model, cfg, all_indices, device)

    # Prepare subject L1 post-ln_f representations as target
    subj_post_lnf = {}
    with torch.no_grad():
        for key, layers in subj_reprs.items():
            vec = layers[1]  # L1 residual
            lnf_vec = ln_f(vec.unsqueeze(0).to(device)).squeeze(0).cpu()
            subj_post_lnf[key] = [lnf_vec]

    results_lines = []

    def log(line=""):
        print(line)
        results_lines.append(line)

    log("Object prediction accuracy when using affine-predicted representations + lm_head")
    log("=" * 78)
    log()

    targets = [
        ("L1 residual (pre-ln_f)", subj_reprs, 1, ln_f),
        ("Attention output (L1)", subj_sublayers["attn"], 0, ln_f),
        ("MLP output (L1)", subj_sublayers["mlp"], 0, ln_f),
        ("Post-ln_f: ln_f(L1 residual)", subj_post_lnf, 0, None),
    ]

    for target_label, target_reprs, target_layer, apply_ln_f in targets:
        log(f"Target: {target_label}")
        log("-" * 50)

        accs = []
        for ri, rkey in enumerate(RELATION_KEYS):
            S = stack_subject_representations(subj_reprs, all_indices, ri, layer=0)
            T = stack_subject_representations(target_reprs, all_indices, ri, layer=target_layer)
            acc = affine_accuracy_per_relation(
                S, T, lm_head_w, apply_ln_f, cfg.vocab, ri, all_indices, device
            )
            log(f"  {rkey}: {acc:.4f}")
            accs.append(acc)

        log(f"  avg: {np.mean(accs):.4f}")
        log()

    # Save
    cfg.train.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = cfg.train.results_dir / "affine_accuracy.md"
    with open(results_path, "w") as f:
        f.write("# Affine Accuracy: Predicting Objects via Affine Transformation + lm_head\n\n")
        f.write("For each (subject, relation) pair:\n")
        f.write("1. Take the L0 embedding sᵢ = token_emb(aᵢ) + pos_emb(2)\n")
        f.write("2. Fit per-relation affine map W, b from L0 to target representation (5-fold CV)\n")
        f.write("3. Compute ô = sᵢ @ W.T + b\n")
        f.write("4. Apply ln_f if target is pre-ln_f, then compute logits = ô @ lm_head.T\n")
        f.write("5. Check if argmax(logits) = correct object token\n\n")
        f.write("```\n")
        f.write("\n".join(results_lines))
        f.write("\n```\n")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
