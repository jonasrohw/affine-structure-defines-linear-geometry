"""
Ravfogel bijective recall with frozen structured embeddings (provably affine).

Embedding construction:
  - Generate 2025 random 32-dim vectors v_0, ..., v_2024
  - Object j: o_j = [v_j; v_j]  (doubled to 64 dims)
  - Subject i: s_i = [v_{π_0(i)}; v_{π_1(i)}]  (first half = rel-0 target, second = rel-1)

The affine structure is guaranteed by construction:
  - Relation 0: extract first 32 dims, duplicate → W_0 = [[I,0],[I,0]]
  - Relation 1: extract second 32 dims, duplicate → W_1 = [[0,I],[0,I]]

The subject/object embeddings and object lm_head rows are FROZEN.
Only the transformer (attention, MLP, LayerNorm) and position/sep/relation
embeddings are learned.

Question: what mechanism does the network learn to implement this affine map?

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/ravfogel_1layer_frozen_affine.py --seed 42
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.affine import fit_affine, reconstruction_accuracy
from src.config import ModelConfig
from src.model import build_model
from src.training import get_lr
from experiments.ravfogel_1layer import (
    RavfogelVocab, generate_permutations, build_sequences,
    SEQ_LENGTH, REL_SENTENCE_LEN,
)

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def build_frozen_embeddings(vocab, dim, seed):
    """Build structured subject/object embeddings.

    Returns:
        base_vectors: (n_entities, dim//2) random vectors
        subject_embs: (n_entities, dim) — [v_{π0(i)}; v_{π1(i)}]
        object_embs: (n_entities, dim) — [v_j; v_j]
    """
    half = dim // 2
    rng = torch.Generator()
    rng.manual_seed(seed + 9999)
    base_vectors = torch.randn(vocab.n_entities, half, generator=rng)
    # Normalize to have consistent scale
    base_vectors = base_vectors / base_vectors.norm(dim=1, keepdim=True) * (half ** 0.5)

    object_embs = torch.cat([base_vectors, base_vectors], dim=1)  # [v_j; v_j]

    subject_embs = torch.zeros(vocab.n_entities, dim)
    for i in range(vocab.n_entities):
        j0 = vocab.relation_object(i, 0)  # π_0(i)
        j1 = vocab.relation_object(i, 1)  # π_1(i)
        subject_embs[i, :half] = base_vectors[j0]
        subject_embs[i, half:] = base_vectors[j1]

    return base_vectors, subject_embs, object_embs


def inject_frozen_embeddings(model, vocab, subject_embs, object_embs):
    """Set subject/object rows in token_emb and lm_head, then freeze them."""
    with torch.no_grad():
        for i in range(vocab.n_entities):
            model.token_emb.weight[vocab.subject_id(i)] = subject_embs[i]
            model.token_emb.weight[vocab.object_id(i)] = object_embs[i]
            model.lm_head.weight[vocab.object_id(i)] = object_embs[i]

    frozen_ids = set()
    for i in range(vocab.n_entities):
        frozen_ids.add(vocab.subject_id(i))
        frozen_ids.add(vocab.object_id(i))

    return frozen_ids


def make_grad_mask_hook(frozen_ids, vocab_size):
    """Creates a hook that zeros gradients for frozen embedding rows."""
    mask = torch.ones(vocab_size, 1)
    for idx in frozen_ids:
        mask[idx] = 0.0

    def hook(grad):
        return grad * mask.to(grad.device)

    return hook


def eval_relation_accuracy(model, vocab, device, batch_size=512):
    model.eval()
    all_pairs = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]
    correct = 0
    total = len(all_pairs)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_pairs = all_pairs[start:start + batch_size]
            contexts = torch.tensor(
                [[vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)]
                 for i, k in batch_pairs],
                dtype=torch.long, device=device,
            )
            logits = model(contexts)
            preds = logits[:, -1].argmax(dim=-1).cpu().tolist()
            for (i, k), pred in zip(batch_pairs, preds):
                correct += int(pred == vocab.object_id(vocab.relation_object(i, k)))
    return correct / total


def train_epoch(model, sequences, optimizer, device, batch_size, lr_fn, global_step):
    x_all = torch.stack([torch.tensor(s[:-1], dtype=torch.long) for s in sequences])
    y_all = torch.stack([torch.tensor(s[1:], dtype=torch.long) for s in sequences])
    dataset = TensorDataset(x_all, y_all)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        current_lr = lr_fn(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
        logits = model(x)
        B, T, V = logits.shape
        loss = nn.functional.cross_entropy(logits.view(B * T, V), y.view(B * T))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
        global_step += 1
    return epoch_loss / max(1, n_batches), global_step


@torch.no_grad()
def extract_hidden_states(model, vocab, device, subject_indices, batch_size=512):
    model.eval()
    keys = []
    contexts = []
    for i in subject_indices:
        for k in range(vocab.num_relations):
            keys.append((i, k))
            contexts.append([vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)])

    all_residuals = {}
    for start in range(0, len(contexts), batch_size):
        batch_ctx = contexts[start:start + batch_size]
        batch_keys = keys[start:start + batch_size]
        batch = torch.tensor(batch_ctx, dtype=torch.long, device=device)
        with model.capture_residuals() as residuals:
            model(batch)
        for layer_idx, layer_res in enumerate(residuals):
            if layer_idx not in all_residuals:
                all_residuals[layer_idx] = {}
            hidden = layer_res[:, 2, :].cpu()
            for i, key in enumerate(batch_keys):
                all_residuals[layer_idx][key] = hidden[i]

    return all_residuals


def affine_analysis(model, vocab, device):
    train_subjs = vocab.train_subject_indices
    test_subjs = vocab.test_subject_indices
    all_subjs = train_subjs + test_subjs

    layer_residuals = extract_hidden_states(model, vocab, device, all_subjs)

    weight = model.lm_head.weight.detach().cpu()
    O_all = weight[vocab.object_offset:vocab.object_offset + vocab.n_entities]

    target = {}
    for i in all_subjs:
        for k in range(vocab.num_relations):
            obj_idx = vocab.relation_object(i, k)
            target[(i, k)] = weight[vocab.object_id(obj_idx)]

    results = {}
    for layer_idx, h_dict in sorted(layer_residuals.items()):
        layer_results = {}
        for k in range(vocab.num_relations):
            S_train = torch.stack([h_dict[(i, k)] for i in train_subjs])
            O_train = torch.stack([target[(i, k)] for i in train_subjs])
            S_test = torch.stack([h_dict[(i, k)] for i in test_subjs])
            O_test = torch.stack([target[(i, k)] for i in test_subjs])

            W, b = fit_affine(S_train, O_train, bias=True)
            train_acc = reconstruction_accuracy(S_train, O_train, W, b, O_all)
            test_acc = reconstruction_accuracy(S_test, O_test, W, b, O_all)

            layer_results[f"r{k}"] = {
                "train_acc": round(train_acc, 4),
                "test_acc": round(test_acc, 4),
            }

        avg_train = np.mean([layer_results[f"r{k}"]["train_acc"] for k in range(vocab.num_relations)])
        avg_test = np.mean([layer_results[f"r{k}"]["test_acc"] for k in range(vocab.num_relations)])
        layer_results["overall"] = {
            "train_acc": round(float(avg_train), 4),
            "test_acc": round(float(avg_test), 4),
        }
        results[f"layer{layer_idx}"] = layer_results

    return results


@torch.no_grad()
def mechanism_analysis(model, vocab, device, base_vectors, batch_size=512):
    """Analyze HOW the network implements the affine map.

    Since the ideal mechanism is:
      R0: extract first 32 dims of subject emb, duplicate
      R1: extract second 32 dims of subject emb, duplicate

    We check:
    1. Attention patterns: what does position 2 (subject) attend to?
    2. MLP contribution: is it ~linear? Does it implement the slot selection?
    3. Residual decomposition: how much comes from embedding vs attention vs MLP?
    4. Per-half analysis: does the output first half match one slot, second half the other?
    """
    model.eval()
    block = model.blocks[0]
    dim = model.model_cfg.hidden_dim
    half = dim // 2

    all_pairs = [(i, k) for i in range(vocab.n_entities) for k in range(vocab.num_relations)]

    # Collect attention patterns and sublayer outputs
    attn_weights_by_rel = {0: [], 1: []}
    h0_by_rel = {0: [], 1: []}
    attn_out_by_rel = {0: [], 1: []}
    mlp_out_by_rel = {0: [], 1: []}
    h_final_by_rel = {0: [], 1: []}

    for start in range(0, len(all_pairs), batch_size):
        batch_pairs = all_pairs[start:start + batch_size]
        contexts = torch.tensor(
            [[vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)]
             for i, k in batch_pairs],
            dtype=torch.long, device=device)

        B, T = contexts.shape
        positions = torch.arange(T, device=device).unsqueeze(0)
        h0 = model.token_emb(contexts) + model.pos_emb(positions)

        # Attention
        ln1_out = block.ln1(h0)
        attn_out = block.attn(ln1_out)
        h1 = h0 + attn_out

        # MLP
        ln2_out = block.ln2(h1)
        mlp_out = block.mlp(ln2_out)
        h2 = h1 + mlp_out

        for idx, (i, k) in enumerate(batch_pairs):
            h0_by_rel[k].append(h0[idx, 2, :].cpu())
            attn_out_by_rel[k].append(attn_out[idx, 2, :].cpu())
            mlp_out_by_rel[k].append(mlp_out[idx, 2, :].cpu())
            h_final_by_rel[k].append(h2[idx, 2, :].cpu())

    results = {}

    # --- Analysis 1: Residual decomposition ---
    for k in range(2):
        H0 = torch.stack(h0_by_rel[k])
        A = torch.stack(attn_out_by_rel[k])
        M = torch.stack(mlp_out_by_rel[k])
        H = torch.stack(h_final_by_rel[k])

        # Variance contribution
        total_var = H.var(dim=0).sum().item()
        h0_var = H0.var(dim=0).sum().item()
        attn_var = A.var(dim=0).sum().item()
        mlp_var = M.var(dim=0).sum().item()

        results[f"r{k}_variance"] = {
            "total": round(total_var, 4),
            "embedding_pct": round(h0_var / total_var * 100, 1),
            "attention_pct": round(attn_var / total_var * 100, 1),
            "mlp_pct": round(mlp_var / total_var * 100, 1),
        }

    # --- Analysis 2: Per-half structure ---
    # For relation 0: ideal output first half = v_{π0(i)}, second half = v_{π0(i)}
    # The subject embedding first half already IS v_{π0(i)}
    # So for R0, the network should "copy first half to second half"
    # For R1, the network should "copy second half to first half"

    ln_f = model.ln_f
    lm_head_weight = model.lm_head.weight.detach().cpu()

    for k in range(2):
        H = torch.stack(h_final_by_rel[k])  # (2025, 64)
        H0 = torch.stack(h0_by_rel[k])

        # After ln_f, what does the output look like per half?
        H_device = H.to(device)
        Z = ln_f(H_device).cpu()

        first_half = Z[:, :half]   # (2025, 32)
        second_half = Z[:, half:]  # (2025, 32)

        # Compare to target: for relation k, target is base_vectors[π_k(i)]
        targets = torch.stack([base_vectors[vocab.relation_object(i, k)]
                               for i in range(vocab.n_entities)])

        # Cosine similarity of each half with target
        cos_first = nn.functional.cosine_similarity(first_half, targets, dim=1).mean().item()
        cos_second = nn.functional.cosine_similarity(second_half, targets, dim=1).mean().item()

        # How much does embedding already contain the answer?
        emb_first = H0[:, :half]
        emb_second = H0[:, half:]
        cos_emb_first = nn.functional.cosine_similarity(emb_first, targets, dim=1).mean().item()
        cos_emb_second = nn.functional.cosine_similarity(emb_second, targets, dim=1).mean().item()

        results[f"r{k}_half_structure"] = {
            "output_first_half_cos_target": round(cos_first, 4),
            "output_second_half_cos_target": round(cos_second, 4),
            "emb_first_half_cos_target": round(cos_emb_first, 4),
            "emb_second_half_cos_target": round(cos_emb_second, 4),
        }

    # --- Analysis 3: MLP linearity per relation ---
    for k in range(2):
        H1 = torch.stack(h0_by_rel[k]) + torch.stack(attn_out_by_rel[k])  # pre-MLP
        M = torch.stack(mlp_out_by_rel[k])

        # Fit affine: H1 → M
        n = H1.shape[0]
        H1_aug = torch.cat([H1, torch.ones(n, 1)], dim=1)
        sol, _, _, _ = torch.linalg.lstsq(H1_aug, M)
        M_pred = H1_aug @ sol
        ss_res = ((M - M_pred) ** 2).sum().item()
        ss_tot = ((M - M.mean(0)) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot

        results[f"r{k}_mlp_linearity"] = {"r2": round(r2, 4)}

    # --- Analysis 4: Does attention contribute relation-specific info? ---
    A0 = torch.stack(attn_out_by_rel[0])
    A1 = torch.stack(attn_out_by_rel[1])
    mean_a0 = A0.mean(0)
    mean_a1 = A1.mean(0)
    all_attn = torch.cat([A0, A1], dim=0)
    grand_mean = all_attn.mean(0)
    total_var = ((all_attn - grand_mean) ** 2).sum().item()
    between_var = (vocab.n_entities * ((mean_a0 - grand_mean) ** 2).sum().item() +
                   vocab.n_entities * ((mean_a1 - grand_mean) ** 2).sum().item())
    results["attention_between_rel_pct"] = round(between_var / total_var * 100, 1)

    cos_rel_dirs = nn.functional.cosine_similarity(
        mean_a0.unsqueeze(0), mean_a1.unsqueeze(0)).item()
    results["attention_rel_means_cos"] = round(cos_rel_dirs, 4)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-entities", type=int, default=2025)
    parser.add_argument("--num-sequences", type=int, default=62500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    vocab = RavfogelVocab(n_entities=args.n_entities, num_relations=2)
    perms = generate_permutations(args.n_entities, 2, seed=args.seed)
    for k, p in enumerate(perms):
        vocab.set_permutation(k, p)

    model_cfg = ModelConfig(n_layers=1, hidden_dim=args.dim, n_heads=args.n_heads)

    print(f"\n{'='*70}")
    print(f"FROZEN AFFINE EMBEDDINGS — 1-LAYER (seed={args.seed})")
    print(f"{'='*70}")
    print(f"Entities: {args.n_entities}, Relations: 2 (random bijections)")
    print(f"Embedding construction:")
    print(f"  Object j: [v_j; v_j]  (doubled 32-dim random vector)")
    print(f"  Subject i: [v_{{π0(i)}}; v_{{π1(i)}}]  (consistent subspace)")
    print(f"  → Affine by construction (R0: extract first half, R1: extract second)")
    print(f"Model: dim={args.dim}, layers=1, heads={args.n_heads}, WITH MLP")
    print(f"Frozen: subject/object embeddings + object lm_head rows")
    print(f"Learned: transformer params, pos emb, sep/relation emb")

    # Build model
    torch.manual_seed(args.seed)
    model = build_model(model_cfg, vocab)

    # Build and inject frozen embeddings
    base_vectors, subject_embs, object_embs = build_frozen_embeddings(
        vocab, args.dim, args.seed)
    frozen_ids = inject_frozen_embeddings(model, vocab, subject_embs, object_embs)

    # Register gradient hooks to freeze specific rows
    hook_emb = model.token_emb.weight.register_hook(
        make_grad_mask_hook(frozen_ids, vocab.vocab_size))
    # Also freeze object rows in lm_head
    obj_ids = {vocab.object_id(j) for j in range(vocab.n_entities)}
    hook_lm = model.lm_head.weight.register_hook(
        make_grad_mask_hook(obj_ids, vocab.vocab_size))

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_frozen = len(frozen_ids) * args.dim + len(obj_ids) * args.dim
    print(f"Model parameters: {n_params:,} (frozen emb rows: {n_frozen:,} values)")

    warmup_steps = 200
    steps_per_epoch = args.num_sequences // args.batch_size
    total_steps = steps_per_epoch * args.n_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    def lr_fn(step):
        return get_lr(step, warmup_steps, total_steps, args.lr)

    print(f"\nTraining: {args.n_epochs} epochs, lr={args.lr}")
    print(f"Steps/epoch: {steps_per_epoch:,}, Total steps: {total_steps:,}")
    print(f"{'='*70}\n", flush=True)

    pre_acc = eval_relation_accuracy(model, vocab, device)
    print(f"Pre-training accuracy: {pre_acc:.3f} (chance: {1/args.n_entities:.4f})\n", flush=True)

    results = {
        "experiment": "ravfogel_1layer_frozen_affine",
        "seed": args.seed,
        "n_entities": args.n_entities,
        "dim": args.dim,
        "n_layers": 1,
        "n_heads": args.n_heads,
        "has_mlp": True,
        "frozen_embeddings": True,
        "n_params": n_params,
        "vocab_size": vocab.vocab_size,
        "n_epochs": args.n_epochs,
        "pre_acc": round(pre_acc, 4),
        "epochs": {},
    }

    rng = random.Random(args.seed + 2000)
    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.n_epochs + 1):
        sequences = build_sequences(vocab, args.num_sequences, SEQ_LENGTH, rng)
        avg_loss, global_step = train_epoch(
            model, sequences, optimizer, device, args.batch_size,
            lr_fn, global_step)
        acc = eval_relation_accuracy(model, vocab, device)
        elapsed = time.time() - t0
        results["epochs"][epoch] = {"loss": round(avg_loss, 4), "accuracy": round(acc, 4)}
        print(f"Ep {epoch:2d}/{args.n_epochs} | loss={avg_loss:.4f} | "
              f"acc={acc:.3f} | {elapsed:.0f}s", flush=True)

    # Affine analysis
    print(f"\n--- Affine analysis ---", flush=True)
    try:
        affine_results = affine_analysis(model, vocab, device)
        for label, res in sorted(affine_results.items()):
            ov = res["overall"]
            print(f"  {label}: train={ov['train_acc']:.3f}  test={ov['test_acc']:.3f}")
        results["affine"] = affine_results
    except Exception as e:
        print(f"  Affine analysis failed: {e}")
        import traceback; traceback.print_exc()
        results["affine"] = str(e)

    # Mechanism analysis
    print(f"\n--- Mechanism analysis ---", flush=True)
    try:
        mech = mechanism_analysis(model, vocab, device, base_vectors)
        print(f"\n  Variance decomposition (at subject position):")
        for k in range(2):
            v = mech[f"r{k}_variance"]
            print(f"    R{k}: emb={v['embedding_pct']:.1f}%  "
                  f"attn={v['attention_pct']:.1f}%  mlp={v['mlp_pct']:.1f}%")

        print(f"\n  Attention between-relation variance: {mech['attention_between_rel_pct']:.1f}%")
        print(f"  Attention relation means cosine: {mech['attention_rel_means_cos']:.4f}")

        print(f"\n  Per-half structure (cos with target v_{{πk(i)}}):")
        for k in range(2):
            h = mech[f"r{k}_half_structure"]
            print(f"    R{k}: output_1st={h['output_first_half_cos_target']:.4f}  "
                  f"output_2nd={h['output_second_half_cos_target']:.4f}")
            print(f"         emb_1st={h['emb_first_half_cos_target']:.4f}  "
                  f"emb_2nd={h['emb_second_half_cos_target']:.4f}")

        print(f"\n  MLP linearity (R² of affine fit input→output):")
        for k in range(2):
            print(f"    R{k}: R²={mech[f'r{k}_mlp_linearity']['r2']:.4f}")

        results["mechanism"] = mech
    except Exception as e:
        print(f"  Mechanism analysis failed: {e}")
        import traceback; traceback.print_exc()
        results["mechanism"] = str(e)

    final_acc = eval_relation_accuracy(model, vocab, device)
    results["final_acc"] = round(final_acc, 4)
    print(f"\n{'='*70}")
    print(f"FINAL accuracy: {final_acc:.3f}")
    print(f"{'='*70}", flush=True)

    # Verify embeddings stayed frozen
    with torch.no_grad():
        emb_drift = 0.0
        for i in range(min(10, vocab.n_entities)):
            actual = model.token_emb.weight[vocab.subject_id(i)].cpu()
            expected = subject_embs[i]
            emb_drift += (actual - expected).norm().item()
        print(f"Embedding drift check (first 10 subjects): {emb_drift:.6f}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ravfogel_1layer_frozen_affine_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    ckpt_dir = ROOT / f"checkpoints_ravfogel_1l_frozen_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "seed": args.seed,
        "permutations": perms,
        "base_vectors": base_vectors,
        "model_cfg": {"n_layers": 1, "hidden_dim": args.dim, "n_heads": args.n_heads},
    }, ckpt_dir / "final.pt")
    print(f"Checkpoint saved to {ckpt_dir / 'final.pt'}", flush=True)

    # Clean up hooks
    hook_emb.remove()
    hook_lm.remove()


if __name__ == "__main__":
    main()
