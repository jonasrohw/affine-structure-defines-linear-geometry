"""
Frozen-embedding experiment: dim=16, 2 relations, 100 subjects.

Tests option (B) vs (C): initialize subject embeddings and object (lm_head)
rows from the affine existence test solution, freeze them, train everything
else. If the transformer reaches ~99% affine accuracy (matching the existence
test), option (B) is ruled out and the answer is (C).

Usage (from salzburg/):
    python experiments/frozen_emb_dim16_2rel.py
"""
import sys
import time
import math
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
from src.training import evaluate, get_lr

RELATION_KEYS = ["R0", "R1"]
ROOT = Path(__file__).parent.parent
DIM = 16
N_SUBJECTS = 100
N_RELATIONS = 2
CKPT_DIR = ROOT / "checkpoints_frozen_dim16_2rel"
RESULTS_DIR = ROOT / "results"


# --------------- Existence test (inline) ---------------

def make_unit_vectors(n, d, seed=42):
    rng = np.random.RandomState(seed)
    O = rng.randn(n, d).astype(np.float32)
    O /= np.linalg.norm(O, axis=1, keepdims=True)
    return torch.from_numpy(O)


def run_existence_test(n, d, n_steps=20000, n_restarts=5, lr=1e-3, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    O = make_unit_vectors(n, d, seed=seed).to(device)

    sigma0 = list(range(n))
    sigma1 = [(i + n // 2) % n for i in range(n)]
    perms = [sigma0, sigma1]
    perm_tensors = [torch.tensor(s, device=device) for s in perms]

    best_loss = float("inf")
    best_state = None

    for restart in range(n_restarts):
        torch.manual_seed(seed + restart * 1000)
        S = torch.randn(n, d, device=device) * 0.1
        S.requires_grad_(True)
        Ws = [torch.randn(d, d, device=device) * 0.1 for _ in range(2)]
        bs = [torch.zeros(d, device=device) for _ in range(2)]
        for w in Ws:
            w.requires_grad_(True)
        for bi in bs:
            bi.requires_grad_(True)

        params = [S] + Ws + bs
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

        for step in range(n_steps):
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=device)
            for r in range(2):
                O_target = O[perm_tensors[r]]
                O_hat = S @ Ws[r].T + bs[r]
                loss = loss + ((O_hat - O_target) ** 2).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

        final_loss = loss.item()
        print(f"    restart {restart} final loss={final_loss:.6f}")
        if final_loss < best_loss:
            best_loss = final_loss
            best_state = (S.detach().clone(), [w.detach().clone() for w in Ws],
                          [bi.detach().clone() for bi in bs])

    S_best, Ws_best, bs_best = best_state

    # Evaluate
    with torch.no_grad():
        for r, rkey in enumerate(RELATION_KEYS):
            target_idx = perm_tensors[r]
            O_hat = S_best @ Ws_best[r].T + bs_best[r]
            cos = F.cosine_similarity(O_hat, O[target_idx], dim=1).mean().item()
            sims = O_hat @ O.T
            preds = sims.argmax(dim=1)
            acc = (preds == target_idx).float().mean().item()
            print(f"    {rkey}: cos={cos:.4f}, Voronoi acc={acc:.4f}")

    return S_best.cpu(), O.cpu(), best_loss


# --------------- Config ---------------

def make_cfg():
    return Config(
        vocab=VocabConfig(n_subjects=N_SUBJECTS, n_objects=N_SUBJECTS, num_relations=N_RELATIONS),
        model=ModelConfig(n_layers=1, hidden_dim=DIM, n_heads=4),
        train=TrainConfig(
            max_epochs=800,
            learning_rate=3e-4,
            batch_size=64,
            seq_length=40,
            num_sequences=5000,
            checkpoint_dir=CKPT_DIR,
            results_dir=RESULTS_DIR,
        ),
        seed=42,
    )


# --------------- Custom training with frozen rows ---------------

def train_frozen(model, train_loader, cfg, device, frozen_subj, frozen_obj,
                 val_loader=None, verbose=True):
    model.to(device)
    model.train()

    subj_offset = cfg.vocab.subject_offset
    obj_offset = cfg.vocab.object_offset
    n = cfg.vocab.n_subjects

    frozen_subj_dev = frozen_subj.to(device)
    frozen_obj_dev = frozen_obj.to(device)

    # Initialize frozen rows
    with torch.no_grad():
        model.token_emb.weight.data[subj_offset:subj_offset + n] = frozen_subj_dev
        model.lm_head.weight.data[obj_offset:obj_offset + n] = frozen_obj_dev

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * cfg.train.max_epochs

    best_val_acc = -1.0
    cfg.train.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = cfg.train.checkpoint_dir / "best.pt"

    global_step = 0
    t0 = time.time()

    for epoch in range(cfg.train.max_epochs):
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            lr = get_lr(global_step, cfg.train.warmup_steps, max_steps, cfg.train.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            logits = model(x)
            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B * T, V), y.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            # Restore frozen rows after each step
            with torch.no_grad():
                model.token_emb.weight.data[subj_offset:subj_offset + n] = frozen_subj_dev
                model.lm_head.weight.data[obj_offset:obj_offset + n] = frozen_obj_dev

            epoch_loss += loss.item()
            global_step += 1

        avg_train_loss = epoch_loss / steps_per_epoch

        if (epoch + 1) % cfg.train.eval_every == 0 or epoch == cfg.train.max_epochs - 1:
            val_metrics = {}
            if val_loader is not None:
                val_metrics = evaluate(model, val_loader, device, cfg)
                if val_metrics["acc_obj"] > best_val_acc:
                    best_val_acc = val_metrics["acc_obj"]
                    torch.save({
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_acc_obj": best_val_acc,
                        "cfg": cfg,
                    }, best_ckpt)

            if verbose and ((epoch + 1) % cfg.train.eval_every == 0 or epoch == cfg.train.max_epochs - 1):
                elapsed = time.time() - t0
                msg = (f"Epoch {epoch+1:4d}/{cfg.train.max_epochs} | "
                       f"train_loss={avg_train_loss:.4f} | lr={lr:.2e} | {elapsed:.0f}s")
                if val_metrics:
                    msg += (f" | val_loss={val_metrics['loss']:.4f}"
                            f" | acc_obj={val_metrics['acc_obj']:.3f}"
                            f" | acc_rel={val_metrics['acc_rel']:.3f}")
                print(msg)

    return best_val_acc


# --------------- Analysis (from dim32_2rel_experiment.py) ---------------

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
    cfg = make_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("FROZEN-EMBEDDING EXPERIMENT: dim=16, 2 relations, 100 subjects")
    print("Tests option (B) vs (C)")
    print("=" * 70)
    print()

    # Phase 1: Run existence test to get embeddings
    print("PHASE 1: Affine existence test (dim=16, n=100, 2 relations)")
    print("-" * 70)
    S_exist, O_exist, exist_loss = run_existence_test(N_SUBJECTS, DIM)
    print(f"  Best loss: {exist_loss:.6f}")
    print()

    # Phase 2: Build model, initialize and freeze, train
    print("PHASE 2: Train transformer with frozen embeddings")
    print("-" * 70)
    torch.manual_seed(cfg.seed)
    train_loader = make_dataloader(cfg, split="train", seed_offset=0)
    model = build_model(cfg.model, cfg.vocab)
    print(f"  Parameters: {model.n_parameters():,}")
    print(f"  Frozen: token_emb subject rows ({N_SUBJECTS}×{DIM}={N_SUBJECTS*DIM})")
    print(f"  Frozen: lm_head object rows ({N_SUBJECTS}×{DIM}={N_SUBJECTS*DIM})")
    print(f"  Total frozen: {2*N_SUBJECTS*DIM:,} values")
    print()

    train_frozen(model, train_loader, cfg, device, S_exist, O_exist,
                 val_loader=train_loader, verbose=True)

    # Load best checkpoint
    best_ckpt = CKPT_DIR / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    train_metrics = evaluate(model, train_loader, device, cfg, max_batches=50)
    print(f"\nFinal: acc_obj={train_metrics['acc_obj']:.4f} | loss={train_metrics['loss']:.4f}")
    print()

    # Verify frozen rows are still correct
    with torch.no_grad():
        subj_rows = model.token_emb.weight.data[cfg.vocab.subject_offset:cfg.vocab.subject_offset + N_SUBJECTS].cpu()
        obj_rows = model.lm_head.weight.data[cfg.vocab.object_offset:cfg.vocab.object_offset + N_SUBJECTS].cpu()
        subj_diff = (subj_rows - S_exist).abs().max().item()
        obj_diff = (obj_rows - O_exist).abs().max().item()
        print(f"Frozen row integrity: max|Δsubj|={subj_diff:.2e}, max|Δobj|={obj_diff:.2e}")
    print()

    # Phase 3: Logit lens
    print("PHASE 3: Logit Lens")
    print("-" * 70)
    ll_accs, ll_total = logit_lens(model, train_loader, device, cfg)
    print(f"  h₀ (embedding):      {ll_accs['h0']:.4f}")
    print(f"  h₀ + attn (pre-MLP): {ll_accs['h0+attn']:.4f}")
    print(f"  h₀ + attn + mlp:     {ll_accs['h0+attn+mlp']:.4f}")
    print()

    # Phase 4: Affine accuracy
    print("PHASE 4: Affine Accuracy")
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
        print(f"    avg: {np.mean(accs):.4f}")
        print()

    # Phase 5: Comparison
    print("PHASE 5: Comparison")
    print("-" * 70)
    print(f"  {'Metric':<35} {'Freely trained':<15} {'Frozen emb':<15} {'Existence':<15}")
    print(f"  {'-'*35} {'-'*15} {'-'*15} {'-'*15}")

    # Freely trained dim=16 results (from dim_sweep_2rel.md)
    free_acc_r0, free_acc_r1 = 1.000, 0.940
    free_lnf_r0, free_lnf_r1 = 0.990, 0.970

    frozen_l1_accs = []
    frozen_lnf_accs = []
    for ri in range(N_RELATIONS):
        a = affine_accuracy_per_relation(
            S0_per_rel[ri], S1_per_rel[ri], lm_head_w, ln_f,
            cfg.vocab, ri, all_indices, device
        )
        frozen_l1_accs.append(a)
        a2 = affine_accuracy_per_relation(
            S0_per_rel[ri], P_lnf_per_rel[ri], lm_head_w, None,
            cfg.vocab, ri, all_indices, device
        )
        frozen_lnf_accs.append(a2)

    print(f"  {'Object accuracy':<35} {'1.000':<15} {train_metrics['acc_obj']:<15.4f} {'—':<15}")
    print(f"  {'Affine acc L1 res R0':<35} {free_acc_r0:<15.3f} {frozen_l1_accs[0]:<15.4f} {'—':<15}")
    print(f"  {'Affine acc L1 res R1':<35} {free_acc_r1:<15.3f} {frozen_l1_accs[1]:<15.4f} {'—':<15}")
    print(f"  {'Affine acc post-ln_f R0':<35} {free_lnf_r0:<15.3f} {frozen_lnf_accs[0]:<15.4f} {'—':<15}")
    print(f"  {'Affine acc post-ln_f R1':<35} {free_lnf_r1:<15.3f} {frozen_lnf_accs[1]:<15.4f} {'—':<15}")
    print(f"  {'Existence Voronoi acc':<35} {'—':<15} {'—':<15} {'0.990':<15}")
    print()

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "frozen_emb_dim16_2rel.md"
    with open(results_path, "w") as f:
        f.write("# Frozen-Embedding Experiment: dim=16, 2 Relations\n\n")
        f.write("## Question\n\n")
        f.write("At dim=16 with 2 relations, the freely trained transformer achieves 97% avg\n")
        f.write("affine accuracy while the existence test achieves 99% Voronoi accuracy.\n")
        f.write("Is the 2pp gap due to (B) architectural limitation or (C) learning?\n\n")
        f.write("## Method\n\n")
        f.write("1. Run existence test → get subject embeddings S and object embeddings O\n")
        f.write("2. Initialize transformer: token_emb subject rows ← S, lm_head object rows ← O\n")
        f.write("3. Freeze those rows (restore after each optimizer step)\n")
        f.write("4. Train everything else (800 epochs, same hyperparameters)\n\n")
        f.write(f"Frozen parameters: {2*N_SUBJECTS*DIM:,} values ")
        f.write(f"({N_SUBJECTS} subjects × {DIM}D + {N_SUBJECTS} objects × {DIM}D)\n\n")
        f.write(f"## Existence Test (dim=16, n=100, 2 relations)\n\n")
        f.write(f"- Best loss: {exist_loss:.6f}\n")
        f.write(f"- Voronoi accuracy: 0.990 (both R0 and R1)\n\n")
        f.write(f"## Training\n\n")
        f.write(f"- Final object accuracy: {train_metrics['acc_obj']:.4f}\n")
        f.write(f"- Final loss: {train_metrics['loss']:.4f}\n\n")
        f.write(f"Frozen row integrity: max|Δsubj|={subj_diff:.2e}, max|Δobj|={obj_diff:.2e}\n\n")
        f.write(f"## Logit Lens\n\n")
        f.write(f"| Stage | Accuracy |\n|-------|----------|\n")
        f.write(f"| h₀ | {ll_accs['h0']:.4f} |\n")
        f.write(f"| h₀ + attn | {ll_accs['h0+attn']:.4f} |\n")
        f.write(f"| h₀ + attn + mlp | {ll_accs['h0+attn+mlp']:.4f} |\n\n")
        f.write("## Affine Accuracy\n\n")
        f.write("| Metric | Freely trained | Frozen emb | Existence test |\n")
        f.write("|--------|---------------|------------|----------------|\n")
        f.write(f"| Object accuracy | 1.000 | {train_metrics['acc_obj']:.4f} | — |\n")
        f.write(f"| Affine acc L1 res R0 | {free_acc_r0:.3f} | {frozen_l1_accs[0]:.4f} | — |\n")
        f.write(f"| Affine acc L1 res R1 | {free_acc_r1:.3f} | {frozen_l1_accs[1]:.4f} | — |\n")
        f.write(f"| Affine acc post-ln_f R0 | {free_lnf_r0:.3f} | {frozen_lnf_accs[0]:.4f} | — |\n")
        f.write(f"| Affine acc post-ln_f R1 | {free_lnf_r1:.3f} | {frozen_lnf_accs[1]:.4f} | — |\n")
        f.write(f"| Affine acc L1 avg | {(free_acc_r0+free_acc_r1)/2:.3f} | {np.mean(frozen_l1_accs):.4f} | — |\n")
        f.write(f"| Affine acc post-ln_f avg | {(free_lnf_r0+free_lnf_r1)/2:.3f} | {np.mean(frozen_lnf_accs):.4f} | — |\n")
        f.write(f"| Existence Voronoi acc | — | — | 0.990 |\n\n")
        f.write("## Interpretation\n\n")
        f.write("(To be filled in after reviewing results.)\n")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
