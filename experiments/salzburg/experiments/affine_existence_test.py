"""
Decoupled affine existence test.

Fix well-separated object embeddings (random unit vectors), then optimize
subject embeddings and per-relation affine maps to minimize squared error.
Tests whether an approximate affine solution EXISTS in low dimensions,
independent of whether a transformer can find it.

Two relations: R0 = identity (σ(i) = i), R1 = cyclic shift by n/2.

Usage (from salzburg/):
    python experiments/affine_existence_test.py
    python experiments/affine_existence_test.py --dim 64 --n_subjects 200
    python experiments/affine_existence_test.py --sweep
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RESULTS_DIR = Path(__file__).parent.parent / "results"


def make_unit_vectors(n, d, seed=42):
    rng = np.random.RandomState(seed)
    O = rng.randn(n, d).astype(np.float32)
    O /= np.linalg.norm(O, axis=1, keepdims=True)
    return torch.from_numpy(O)


def evaluate(S, Ws, bs, O, perms, device):
    results = {}
    with torch.no_grad():
        for r, sigma in enumerate(perms):
            target_idx = torch.tensor(sigma, device=device)
            O_target = O[target_idx]
            O_hat = S @ Ws[r].T + bs[r]

            mse = ((O_hat - O_target) ** 2).mean().item()
            cos = F.cosine_similarity(O_hat, O_target, dim=1).mean().item()

            sims = O_hat @ O.T
            preds = sims.argmax(dim=1)
            acc = (preds == target_idx).float().mean().item()

            results[f"R{r}"] = {"mse": mse, "cos": cos, "acc": acc}
    return results


def optimize(n, d, n_steps=20000, n_restarts=5, lr=1e-3, seed=42, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    O = make_unit_vectors(n, d, seed=seed).to(device)

    with torch.no_grad():
        dists = torch.cdist(O, O)
        dists.fill_diagonal_(float("inf"))
        min_dist = dists.min().item()
        avg_dist = dists[dists != float("inf")].mean().item()

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

            if verbose and step % 5000 == 0:
                print(f"    restart {restart}, step {step:5d} | loss={loss.item():.6f}")

        final_loss = loss.item()
        if verbose:
            print(f"    restart {restart} final loss={final_loss:.6f}")
        if final_loss < best_loss:
            best_loss = final_loss
            best_state = (
                S.detach().clone(),
                [w.detach().clone() for w in Ws],
                [bi.detach().clone() for bi in bs],
            )

    S, Ws, bs = best_state
    results = evaluate(S, Ws, bs, O, perms, device)

    return {
        "loss": best_loss,
        "min_obj_dist": min_dist,
        "avg_obj_dist": avg_dist,
        "relations": results,
        "n": n,
        "d": d,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--n_subjects", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--n_restarts", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if args.sweep:
        configs = [
            (32, 50),
            (32, 100),
            (32, 200),
            (32, 400),
            (64, 100),
            (64, 200),
            (64, 400),
            (128, 100),
            (128, 400),
        ]
        print("Affine existence test: decoupled (fixed unit-vector objects)")
        print("Relations: R0=identity, R1=shift by n/2")
        print(f"Steps: {args.n_steps}, Restarts: {args.n_restarts}")
        print()
        header = f"{'dim':>5} {'n':>5} | {'R0 cos':>8} {'R0 acc':>8} | {'R1 cos':>8} {'R1 acc':>8} | {'min_dist':>9}"
        print(header)
        print("-" * len(header))

        all_results = []
        for d, n in configs:
            res = optimize(
                n, d, n_steps=args.n_steps, n_restarts=args.n_restarts, lr=args.lr
            )
            r0 = res["relations"]["R0"]
            r1 = res["relations"]["R1"]
            print(
                f"{d:5d} {n:5d} | {r0['cos']:8.4f} {r0['acc']:8.4f} "
                f"| {r1['cos']:8.4f} {r1['acc']:8.4f} | {res['min_obj_dist']:9.4f}"
            )
            all_results.append(res)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = RESULTS_DIR / "affine_existence_test.md"
        with open(results_path, "w") as f:
            f.write("# Affine Existence Test (Decoupled)\n\n")
            f.write("**Question**: given well-separated object embeddings, do subject\n")
            f.write("embeddings and affine maps exist that achieve low approximation error?\n\n")
            f.write("**Method**: fix O as random unit vectors, optimize S, W_r, b_r to minimize\n")
            f.write("Σ_r Σ_i ||W_r s_i + b_r − o_{σ_r(i)}||². Two relations:\n")
            f.write("R0 = identity, R1 = cyclic shift by n/2.\n\n")
            f.write(f"Optimization: Adam, {args.n_steps} steps, {args.n_restarts} restarts, best kept.\n\n")
            f.write("## Results\n\n")
            f.write("| dim | n | R0 cos(o,ô) | R0 Voronoi acc | R1 cos(o,ô) | R1 Voronoi acc | min obj dist |\n")
            f.write("|-----|---|-------------|----------------|-------------|----------------|--------------|\n")
            for res in all_results:
                r0 = res["relations"]["R0"]
                r1 = res["relations"]["R1"]
                f.write(
                    f"| {res['d']} | {res['n']} "
                    f"| {r0['cos']:.4f} | {r0['acc']:.4f} "
                    f"| {r1['cos']:.4f} | {r1['acc']:.4f} "
                    f"| {res['min_obj_dist']:.4f} |\n"
                )
            f.write("\n")
        print(f"\nSaved to {results_path}")

    else:
        print(f"Affine existence test: dim={args.dim}, n_subjects={args.n_subjects}")
        print(f"Relations: R0=identity, R1=shift by {args.n_subjects // 2}")
        print(f"Steps: {args.n_steps}, Restarts: {args.n_restarts}")
        print()
        res = optimize(
            args.n_subjects,
            args.dim,
            args.n_steps,
            args.n_restarts,
            args.lr,
            verbose=True,
        )
        print()
        print(
            f"Object embeddings: min pairwise dist={res['min_obj_dist']:.4f}, "
            f"avg={res['avg_obj_dist']:.4f}"
        )
        print(f"Best loss: {res['loss']:.6f}")
        print()
        for rkey, rval in res["relations"].items():
            print(
                f"  {rkey}: MSE={rval['mse']:.6f} | cos(o,ô)={rval['cos']:.4f} "
                f"| Voronoi acc={rval['acc']:.4f}"
            )


if __name__ == "__main__":
    main()
