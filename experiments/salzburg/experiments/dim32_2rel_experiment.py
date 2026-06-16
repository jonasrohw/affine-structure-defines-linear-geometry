"""
Run the full experiment pipeline with hidden_dim=32 and 2 relations.

Matches the affine existence test setting: R0=identity, R1=shift by n/2.
Baseline for comparing transformer behavior with the existence test results.

Usage (from salzburg/):
    python experiments/dim32_2rel_experiment.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
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
from src.training import evaluate, load_checkpoint, train

RELATION_KEYS = ["R0", "R1"]
ROOT = Path(__file__).parent.parent
DIM = 32
N_SUBJECTS = 100
N_RELATIONS = 2
CKPT_DIR = ROOT / "checkpoints_dim32_2rel"
RESULTS_DIR = ROOT / "results"


def make_cfg():
    return Config(
        vocab=VocabConfig(n_subjects=N_SUBJECTS, n_objects=N_SUBJECTS, num_relations=N_RELATIONS),
        model=ModelConfig(
            n_layers=1,
            hidden_dim=DIM,
            n_heads=4,
        ),
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


def train_model(cfg, device):
    torch.manual_seed(cfg.seed)
    train_loader = make_dataloader(cfg, split="train", seed_offset=0)
    model = build_model(cfg.model, cfg.vocab)
    print(f"Parameters: {model.n_parameters():,}")
    print(f"Hidden dim: {DIM}, Subjects: {N_SUBJECTS}, Relations: {N_RELATIONS}")
    print(f"Vocab size: {cfg.vocab.vocab_size}")
    print(f"Facts: {N_RELATIONS}×{N_SUBJECTS}={N_RELATIONS * N_SUBJECTS}\n")

    train(
        model=model,
        train_loader=train_loader,
        cfg=cfg,
        device=device,
        val_loader=train_loader,
        verbose=True,
    )

    best_ckpt = CKPT_DIR / "best.pt"
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model, device)
    model.to(device)
    model.eval()

    train_metrics = evaluate(model, train_loader, device, cfg, max_batches=50)
    print(f"\nFinal: acc_obj={train_metrics['acc_obj']:.4f} | loss={train_metrics['loss']:.4f}")
    return model, train_metrics


def fmt3(dcos, co, cs):
    return f"Δcos={dcos:+.3f}  cos(o,ô)={co:+.3f}  cos(o,s)={cs:+.3f}"


def delta_cos_decomposed(S, O, W, b):
    O_hat = S @ W.T
    if b is not None:
        O_hat = O_hat + b
    o_norm = F.normalize(O, dim=1)
    o_hat_norm = F.normalize(O_hat, dim=1)
    s_norm = F.normalize(S, dim=1)
    cos_o_ohat = (o_norm * o_hat_norm).sum(dim=1)
    cos_o_s = (o_norm * s_norm).sum(dim=1)
    dcos = (cos_o_ohat - cos_o_s).mean().item()
    return dcos, cos_o_ohat.mean().item(), cos_o_s.mean().item()


def kfold_decomposed(S, O, k=5):
    n = S.shape[0]
    fold_size = n // k
    indices = np.arange(n)
    all_dcos, all_co, all_cs = [], [], []
    for fold in range(k):
        val_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        W, b = fit_affine(S[train_idx], O[train_idx], bias=True)
        dc, co, cs = delta_cos_decomposed(S[val_idx], O[val_idx], W, b)
        all_dcos.append(dc)
        all_co.append(co)
        all_cs.append(cs)
    return np.mean(all_dcos), np.mean(all_co), np.mean(all_cs)


def affine_accuracy_per_relation(S, target, lm_head_w, ln_f, vocab, relation_idx, subject_indices, device, k=5):
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


def main():
    cfg = make_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"=" * 70)
    print(f"SALZBURG dim=32, 2 RELATIONS EXPERIMENT")
    print(f"100 subjects, 2 relations (R0=identity, R1=shift by 50)")
    print(f"2×100=200 facts")
    print(f"=" * 70)
    print()

    # --- Train ---
    print("PHASE 1: Training")
    print("-" * 70)
    model, train_metrics = train_model(cfg, device)
    print()

    all_indices = list(range(cfg.vocab.n_subjects))
    n_rel = cfg.vocab.num_relations

    # --- Logit Lens ---
    print("PHASE 2: Logit Lens")
    print("-" * 70)
    loader = make_dataloader(cfg, split="train", seed_offset=0)
    ll_accs, ll_total = logit_lens(model, loader, device, cfg)
    print(f"  h₀ (embedding):      {ll_accs['h0']:.4f}")
    print(f"  h₀ + attn (pre-MLP): {ll_accs['h0+attn']:.4f}")
    print(f"  h₀ + attn + mlp:     {ll_accs['h0+attn+mlp']:.4f}")
    print()

    # --- Δcos decomposition ---
    print("PHASE 3: Δcos Decomposition")
    print("-" * 70)
    print("Extracting representations...")
    subj_reprs = get_subject_representations(model, cfg, all_indices, device)
    subj_sublayers = get_subject_sublayer_outputs(model, cfg, all_indices, device)

    subj_post_lnf = {}
    ln_f = model.ln_f
    with torch.no_grad():
        for key, layers in subj_reprs.items():
            vec = layers[1]
            lnf_vec = ln_f(vec.unsqueeze(0).to(device)).squeeze(0).cpu()
            subj_post_lnf[key] = [lnf_vec]

    S0_per_rel = [stack_subject_representations(subj_reprs, all_indices, ri, layer=0) for ri in range(n_rel)]
    S1_per_rel = [stack_subject_representations(subj_reprs, all_indices, ri, layer=1) for ri in range(n_rel)]
    P_lnf_per_rel = [stack_subject_representations(subj_post_lnf, all_indices, ri, layer=0) for ri in range(n_rel)]
    attn_per_rel = [stack_subject_representations(subj_sublayers["attn"], all_indices, ri, layer=0) for ri in range(n_rel)]
    mlp_per_rel = [stack_subject_representations(subj_sublayers["mlp"], all_indices, ri, layer=0) for ri in range(n_rel)]

    dcos_lines = []

    def log(line=""):
        print(line)
        dcos_lines.append(line)

    targets = [
        ("L1 residual (pre-ln_f)", S1_per_rel),
        ("Attention output (L1)", attn_per_rel),
        ("MLP output (L1)", mlp_per_rel),
        ("Post-ln_f: ln_f(L1 residual)", P_lnf_per_rel),
    ]

    for label, T_per_rel in targets:
        log(f"  Target: {label}")
        for ri, rkey in enumerate(RELATION_KEYS):
            dc, co, cs = kfold_decomposed(S0_per_rel[ri], T_per_rel[ri])
            log(f"    {rkey}: {fmt3(dc, co, cs)}")
        dcs, cos_list, css = [], [], []
        for ri in range(n_rel):
            dc, co, cs = kfold_decomposed(S0_per_rel[ri], T_per_rel[ri])
            dcs.append(dc); cos_list.append(co); css.append(cs)
        log(f"    avg: {fmt3(np.mean(dcs), np.mean(cos_list), np.mean(css))}")
        log()

    # --- Affine Accuracy ---
    print("PHASE 4: Affine Accuracy")
    print("-" * 70)
    lm_head_w = model.lm_head.weight.detach().cpu()

    acc_targets = [
        ("L1 residual (pre-ln_f)", S1_per_rel, ln_f),
        ("MLP output (L1)", mlp_per_rel, ln_f),
        ("Post-ln_f: ln_f(L1 residual)", P_lnf_per_rel, None),
    ]

    acc_lines = []
    for label, T_per_rel, apply_ln_f in acc_targets:
        line = f"  Target: {label}"
        print(line); acc_lines.append(line)
        accs = []
        for ri, rkey in enumerate(RELATION_KEYS):
            acc = affine_accuracy_per_relation(
                S0_per_rel[ri], T_per_rel[ri], lm_head_w, apply_ln_f,
                cfg.vocab, ri, all_indices, device
            )
            line = f"    {rkey}: {acc:.4f}"
            print(line); acc_lines.append(line)
            accs.append(acc)
        line = f"    avg: {np.mean(accs):.4f}"
        print(line); acc_lines.append(line)
        acc_lines.append("")
        print()

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "dim32_2rel_experiment.md"
    with open(results_path, "w") as f:
        f.write("# Salzburg dim=32, 2 Relations Experiment\n\n")
        f.write("100 subjects, 2 relations (R0=identity, R1=shift by 50) — 200 facts.\n")
        f.write("Baseline for the option (B) test: matches the affine existence test setting.\n\n")
        f.write(f"## Training (800 epochs)\n\n")
        f.write(f"- Final object accuracy: {train_metrics['acc_obj']:.4f}\n")
        f.write(f"- Final loss: {train_metrics['loss']:.4f}\n")
        f.write(f"- Model parameters: {model.n_parameters():,}\n\n")
        f.write(f"## Logit Lens (n={ll_total} object positions)\n\n")
        f.write(f"| Stage | Object Accuracy |\n")
        f.write(f"|-------|-----------------|\n")
        f.write(f"| h₀ (embedding) | {ll_accs['h0']:.4f} |\n")
        f.write(f"| h₀ + attn (pre-MLP) | {ll_accs['h0+attn']:.4f} |\n")
        f.write(f"| h₀ + attn + mlp (full) | {ll_accs['h0+attn+mlp']:.4f} |\n\n")
        f.write("## Δcos Decomposition (L0 → target, per-relation, natural context)\n\n")
        f.write("```\n")
        f.write("\n".join(dcos_lines))
        f.write("\n```\n\n")
        f.write("## Affine Accuracy (L0 → target → lm_head)\n\n")
        f.write("```\n")
        f.write("\n".join(acc_lines))
        f.write("\n```\n\n")
        f.write("## Comparison with 3-relation experiment (dim=32, 100 subjects)\n\n")
        f.write("| Metric | 3 relations | 2 relations |\n")
        f.write("|--------|------------|-------------|\n")
        f.write(f"| Object accuracy | 1.0000 | {train_metrics['acc_obj']:.4f} |\n")
        f.write(f"| Logit lens h₀+attn | 0.4343 | {ll_accs['h0+attn']:.4f} |\n")
        dcos_avg = np.mean([kfold_decomposed(S0_per_rel[ri], S1_per_rel[ri])[1] for ri in range(n_rel)])
        f.write(f"| cos(o,ô) L1 residual avg | 0.733 | {dcos_avg:.3f} |\n")
        acc_avg = np.mean([
            affine_accuracy_per_relation(
                S0_per_rel[ri], S1_per_rel[ri], lm_head_w, ln_f,
                cfg.vocab, ri, all_indices, device
            ) for ri in range(n_rel)
        ])
        f.write(f"| Affine accuracy (L1 res) avg | 0.473 | {acc_avg:.3f} |\n")
        acc_lnf_avg = np.mean([
            affine_accuracy_per_relation(
                S0_per_rel[ri], P_lnf_per_rel[ri], lm_head_w, None,
                cfg.vocab, ri, all_indices, device
            ) for ri in range(n_rel)
        ])
        f.write(f"| Affine accuracy (post-ln_f) avg | 0.463 | {acc_lnf_avg:.3f} |\n")
        f.write(f"| Facts | 300 | 200 |\n")
        f.write(f"| Existence test Voronoi acc | — | 1.000 |\n")
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
