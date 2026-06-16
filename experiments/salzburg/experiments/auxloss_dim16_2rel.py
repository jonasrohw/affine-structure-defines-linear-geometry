"""
Auxiliary affine loss experiment: dim=16, 2 relations, 100 subjects.

Tests option (C): add an auxiliary loss that encourages the L0→L1 mapping
to be affine. If this pushes affine accuracy toward the existence test
ceiling (99%), then the architecture can implement it and the issue was
that SGD had no gradient pressure toward affine structure.

Auxiliary parameters W_r, b_r (one affine map per relation) are jointly
optimized. Loss = CE + λ · Σ_r mean||W_r · h₀ + b_r − h₁||².

Usage (from salzburg/):
    python experiments/auxloss_dim16_2rel.py
    python experiments/auxloss_dim16_2rel.py --lam 0.5
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.affine import fit_affine
from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.data_generation import make_dataloader, token_type_mask
from src.model import build_model
from src.representations import (
    get_subject_representations,
    get_subject_sublayer_outputs,
    stack_subject_representations,
)
from src.training import evaluate, get_lr, load_checkpoint

RELATION_KEYS = ["R0", "R1"]
ROOT = Path(__file__).parent.parent
DIM = 16
N_SUBJECTS = 100
N_RELATIONS = 2
RESULTS_DIR = ROOT / "results"


def make_cfg(lam):
    ckpt_dir = ROOT / f"checkpoints_auxloss_dim16_2rel_lam{lam}"
    return Config(
        vocab=VocabConfig(n_subjects=N_SUBJECTS, n_objects=N_SUBJECTS, num_relations=N_RELATIONS),
        model=ModelConfig(n_layers=1, hidden_dim=DIM, n_heads=4),
        train=TrainConfig(
            max_epochs=800,
            learning_rate=3e-4,
            batch_size=64,
            seq_length=40,
            num_sequences=5000,
            checkpoint_dir=ckpt_dir,
            results_dir=RESULTS_DIR,
        ),
        seed=42,
    )


def train_with_aux_loss(model, train_loader, cfg, device, lam, val_loader=None, verbose=True):
    model.to(device)
    model.train()

    vocab = cfg.vocab
    n_rel = vocab.num_relations

    # Auxiliary affine parameters: one (W, b) per relation
    aux_W = [nn.Parameter(torch.randn(DIM, DIM, device=device) * 0.02) for _ in range(n_rel)]
    aux_b = [nn.Parameter(torch.zeros(DIM, device=device)) for _ in range(n_rel)]

    all_params = list(model.parameters()) + aux_W + aux_b
    optimizer = torch.optim.AdamW(all_params, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * cfg.train.max_epochs

    best_val_acc = -1.0
    cfg.train.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = cfg.train.checkpoint_dir / "best.pt"

    global_step = 0
    t0 = time.time()

    for epoch in range(cfg.train.max_epochs):
        epoch_ce = 0.0
        epoch_aux = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            B, T = x.shape

            lr = get_lr(global_step, cfg.train.warmup_steps, max_steps, cfg.train.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Manual forward pass to get h₀ and h₁ with gradients
            positions = torch.arange(T, device=device).unsqueeze(0)
            h0 = model.token_emb(x) + model.pos_emb(positions)
            block = model.blocks[0]
            attn_out = block.attn(block.ln1(h0))
            h0_plus_attn = h0 + attn_out
            mlp_out = block.mlp(block.ln2(h0_plus_attn))
            h1 = h0_plus_attn + mlp_out
            logits = model.lm_head(model.ln_f(h1))

            # CE loss
            V = logits.shape[-1]
            ce_loss = nn.functional.cross_entropy(logits.view(B * T, V), y.view(B * T))

            # Auxiliary affine loss on subject positions
            subj_mask = (x >= vocab.subject_offset) & (x < vocab.object_offset)
            subj_positions = subj_mask.nonzero()  # (N, 2)

            if subj_positions.shape[0] > 0:
                batch_idx = subj_positions[:, 0]
                pos_idx = subj_positions[:, 1]

                rel_tokens = x[batch_idx, pos_idx - 1]
                rel_indices = rel_tokens - vocab.relation_offset

                h0_subj = h0[batch_idx, pos_idx]
                h1_subj = h1[batch_idx, pos_idx]

                aux_loss = torch.tensor(0.0, device=device)
                for r in range(n_rel):
                    mask_r = (rel_indices == r)
                    if mask_r.sum() == 0:
                        continue
                    h0_r = h0_subj[mask_r]
                    h1_r = h1_subj[mask_r]
                    h1_hat = h0_r @ aux_W[r].T + aux_b[r]
                    aux_loss = aux_loss + ((h1_hat - h1_r) ** 2).mean()
            else:
                aux_loss = torch.tensor(0.0, device=device)

            loss = ce_loss + lam * aux_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, cfg.train.grad_clip)
            optimizer.step()

            epoch_ce += ce_loss.item()
            epoch_aux += aux_loss.item()
            n_batches += 1
            global_step += 1

        avg_ce = epoch_ce / n_batches
        avg_aux = epoch_aux / n_batches

        if (epoch + 1) % cfg.train.eval_every == 0 or epoch == cfg.train.max_epochs - 1:
            val_metrics = {}
            if val_loader is not None:
                val_metrics = evaluate(model, val_loader, device, cfg)
                if val_metrics["acc_obj"] > best_val_acc:
                    best_val_acc = val_metrics["acc_obj"]
                    torch.save({
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "aux_W": [w.data.clone() for w in aux_W],
                        "aux_b": [b.data.clone() for b in aux_b],
                        "val_acc_obj": best_val_acc,
                        "cfg": cfg,
                    }, best_ckpt)

            if verbose:
                elapsed = time.time() - t0
                msg = (f"Epoch {epoch+1:4d}/{cfg.train.max_epochs} | "
                       f"CE={avg_ce:.4f} aux={avg_aux:.4f} | lr={lr:.2e} | {elapsed:.0f}s")
                if val_metrics:
                    msg += (f" | val_loss={val_metrics['loss']:.4f}"
                            f" | acc_obj={val_metrics['acc_obj']:.3f}")
                print(msg)

    return best_val_acc


# --------------- Analysis ---------------

def affine_accuracy_per_relation(S, target, lm_head_w, ln_f, vocab, relation_idx,
                                  subject_indices, device, k=5):
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
            logits = O_hat @ lm_head_w.T
            preds = logits.argmax(dim=1)
        for idx_in_fold, global_idx in enumerate(val_idx):
            subj_idx = subject_indices[global_idx]
            expected_obj = vocab.apply_relation(relation_idx, subj_idx)
            expected_token = vocab.object_id(expected_obj)
            if preds[idx_in_fold].item() == expected_token:
                correct += 1
            total += 1
    return correct / max(1, total)


def kfold_cos(S, O, k=5):
    n = S.shape[0]
    fold_size = n // k
    indices = np.arange(n)
    all_cos = []
    for fold in range(k):
        val_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        W, b = fit_affine(S[train_idx], O[train_idx], bias=True)
        O_hat = S[val_idx] @ W.T
        if b is not None:
            O_hat = O_hat + b
        cos = F.cosine_similarity(O_hat, O[val_idx], dim=1).mean().item()
        all_cos.append(cos)
    return np.mean(all_cos)


@torch.no_grad()
def logit_lens(model, loader, device, cfg, max_batches=50):
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


# --------------- Main ---------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, default=1.0)
    args = parser.parse_args()
    lam = args.lam

    cfg = make_cfg(lam)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"AUXILIARY AFFINE LOSS: dim={DIM}, 2 relations, 100 subjects, λ={lam}")
    print("Loss = CE + λ · Σ_r mean||W_r · h₀ + b_r − h₁||²")
    print("=" * 70)
    print()

    # Train
    print("PHASE 1: Training with auxiliary loss")
    print("-" * 70)
    torch.manual_seed(cfg.seed)
    train_loader = make_dataloader(cfg, split="train", seed_offset=0)
    model = build_model(cfg.model, cfg.vocab)
    print(f"  Parameters: {model.n_parameters():,} (model) + {N_RELATIONS * (DIM*DIM + DIM)} (auxiliary)")
    print()

    train_with_aux_loss(model, train_loader, cfg, device, lam,
                        val_loader=train_loader, verbose=True)

    # Load best
    best_ckpt = cfg.train.checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model, device)
    model.to(device)
    model.eval()

    train_metrics = evaluate(model, train_loader, device, cfg, max_batches=50)
    print(f"\nFinal: acc_obj={train_metrics['acc_obj']:.4f} | loss={train_metrics['loss']:.4f}")
    print()

    # Logit lens
    print("PHASE 2: Logit Lens")
    print("-" * 70)
    ll_accs, ll_total = logit_lens(model, train_loader, device, cfg)
    print(f"  h₀ (embedding):      {ll_accs['h0']:.4f}")
    print(f"  h₀ + attn (pre-MLP): {ll_accs['h0+attn']:.4f}")
    print(f"  h₀ + attn + mlp:     {ll_accs['h0+attn+mlp']:.4f}")
    print()

    # Affine accuracy
    print("PHASE 3: Affine Accuracy")
    print("-" * 70)
    all_indices = list(range(N_SUBJECTS))
    subj_reprs = get_subject_representations(model, cfg, all_indices, device)

    subj_post_lnf = {}
    ln_f = model.ln_f
    with torch.no_grad():
        for key, layers in subj_reprs.items():
            vec = layers[1]
            lnf_vec = ln_f(vec.unsqueeze(0).to(device)).squeeze(0).cpu()
            subj_post_lnf[key] = [lnf_vec]

    S0_per_rel = [stack_subject_representations(subj_reprs, all_indices, ri, layer=0) for ri in range(N_RELATIONS)]
    S1_per_rel = [stack_subject_representations(subj_reprs, all_indices, ri, layer=1) for ri in range(N_RELATIONS)]
    P_lnf_per_rel = [stack_subject_representations(subj_post_lnf, all_indices, ri, layer=0) for ri in range(N_RELATIONS)]

    lm_head_w = model.lm_head.weight.detach().cpu()

    targets = [
        ("L1 residual (pre-ln_f)", S1_per_rel, ln_f),
        ("Post-ln_f", P_lnf_per_rel, None),
    ]

    all_accs = {}
    for label, T_per_rel, apply_ln_f in targets:
        print(f"  Target: {label}")
        accs = []
        for ri, rkey in enumerate(RELATION_KEYS):
            acc = affine_accuracy_per_relation(
                S0_per_rel[ri], T_per_rel[ri], lm_head_w, apply_ln_f,
                cfg.vocab, ri, all_indices, device
            )
            cos = kfold_cos(S0_per_rel[ri], T_per_rel[ri])
            print(f"    {rkey}: affine_acc={acc:.4f}, cos(o,ô)={cos:.4f}")
            accs.append(acc)
        avg = np.mean(accs)
        print(f"    avg: {avg:.4f}")
        all_accs[label] = {"per_rel": accs, "avg": avg}
        print()

    # Comparison
    print("PHASE 4: Comparison")
    print("-" * 70)
    print(f"  {'Metric':<35} {'No aux (baseline)':<18} {'With aux (λ={lam})':<18} {'Existence':<12}")
    print(f"  {'-'*35} {'-'*18} {'-'*18} {'-'*12}")

    free_l1 = [1.000, 0.940]
    free_lnf = [0.990, 0.970]
    aux_l1 = all_accs["L1 residual (pre-ln_f)"]["per_rel"]
    aux_lnf = all_accs["Post-ln_f"]["per_rel"]

    print(f"  {'Object accuracy':<35} {'1.000':<18} {train_metrics['acc_obj']:<18.4f} {'—':<12}")
    print(f"  {'Affine acc L1 res R0':<35} {free_l1[0]:<18.3f} {aux_l1[0]:<18.4f} {'—':<12}")
    print(f"  {'Affine acc L1 res R1':<35} {free_l1[1]:<18.3f} {aux_l1[1]:<18.4f} {'—':<12}")
    print(f"  {'Affine acc L1 avg':<35} {np.mean(free_l1):<18.3f} {np.mean(aux_l1):<18.4f} {'—':<12}")
    print(f"  {'Affine acc post-ln_f R0':<35} {free_lnf[0]:<18.3f} {aux_lnf[0]:<18.4f} {'—':<12}")
    print(f"  {'Affine acc post-ln_f R1':<35} {free_lnf[1]:<18.3f} {aux_lnf[1]:<18.4f} {'—':<12}")
    print(f"  {'Affine acc post-ln_f avg':<35} {np.mean(free_lnf):<18.3f} {np.mean(aux_lnf):<18.4f} {'—':<12}")
    print(f"  {'Existence Voronoi acc':<35} {'—':<18} {'—':<18} {'0.990':<12}")
    print()

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"auxloss_dim16_2rel_lam{lam}.md"
    with open(results_path, "w") as f:
        f.write(f"# Auxiliary Affine Loss: dim=16, 2 Relations, λ={lam}\n\n")
        f.write("## Question\n\n")
        f.write("Is the gap between transformer affine accuracy (97%) and the existence\n")
        f.write("test ceiling (99%) due to (B) architectural limitation or (C) learning?\n\n")
        f.write("## Method\n\n")
        f.write("Add an auxiliary loss that encourages the L0→L1 mapping to be affine:\n\n")
        f.write("    Loss = CE + λ · Σ_r mean||W_r · h₀ + b_r − h₁||²\n\n")
        f.write("W_r, b_r are auxiliary learnable parameters (one affine map per relation),\n")
        f.write("optimized jointly with the transformer. The CE loss ensures correct\n")
        f.write("predictions; the auxiliary loss provides gradient pressure toward affine\n")
        f.write("structure in the internal representations.\n\n")
        f.write(f"λ = {lam}\n\n")
        f.write("## Training\n\n")
        f.write(f"- Final object accuracy: {train_metrics['acc_obj']:.4f}\n")
        f.write(f"- Final loss (CE only): {train_metrics['loss']:.4f}\n\n")
        f.write("## Logit Lens\n\n")
        f.write("| Stage | No aux | With aux |\n|-------|--------|----------|\n")
        free_ll = {"h0": 0.259, "h0+attn": 0.460, "h0+attn+mlp": 1.000}
        for s in ["h0", "h0+attn", "h0+attn+mlp"]:
            f.write(f"| {s} | {free_ll[s]:.3f} | {ll_accs[s]:.4f} |\n")
        f.write("\n")
        f.write("## Affine Accuracy\n\n")
        f.write(f"| Metric | No aux (baseline) | With aux (λ={lam}) | Existence |\n")
        f.write("|--------|------------------|-------------------|------------|\n")
        f.write(f"| Object accuracy | 1.000 | {train_metrics['acc_obj']:.4f} | — |\n")
        f.write(f"| Affine acc L1 res R0 | {free_l1[0]:.3f} | {aux_l1[0]:.4f} | — |\n")
        f.write(f"| Affine acc L1 res R1 | {free_l1[1]:.3f} | {aux_l1[1]:.4f} | — |\n")
        f.write(f"| Affine acc L1 avg | {np.mean(free_l1):.3f} | {np.mean(aux_l1):.4f} | — |\n")
        f.write(f"| Affine acc post-ln_f R0 | {free_lnf[0]:.3f} | {aux_lnf[0]:.4f} | — |\n")
        f.write(f"| Affine acc post-ln_f R1 | {free_lnf[1]:.3f} | {aux_lnf[1]:.4f} | — |\n")
        f.write(f"| Affine acc post-ln_f avg | {np.mean(free_lnf):.3f} | {np.mean(aux_lnf):.4f} | — |\n")
        f.write(f"| Existence Voronoi acc | — | — | 0.990 |\n\n")
        f.write("## Interpretation\n\n")
        f.write("(To be filled in after reviewing results.)\n")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
