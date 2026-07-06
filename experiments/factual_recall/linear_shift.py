"""Linear vs nonlinear relation representations, with one probe metric.

Two synthetic worlds, one training protocol, one metric.

Worlds
------
``category`` (many-to-one, used for the matched-frequency contrast):
  - 192 subjects ``a_i`` with attributes ``(f_{i mod 32}, g_{i div 32})``:
    32 f-values x 6 g-values, each subject a unique (f, g) pair.
  - 32 objects ``b_j`` with attributes ``(f_{(j+3) mod 32}, g_0)``.
  - Semantic relation: ``R0(a_i) = b_{f(i)}`` -- the target is the object
    whose f attribute is the subject's f attribute shifted by 3.
  - Random relation: ``R1(a_i) = b_{rho(i)}`` -- balanced random assignment
    onto the same 32 objects, independent of attributes.
  - Split: per f-bucket, 4 trained + 2 held-out subjects (128 / 64). rho is
    balanced within both groups, so the two relations have identical
    frequency statistics at every level (triples, subjects, objects).

``onetoone`` (one-to-one, used for the single-occurrence phase shift):
  - 1536 subjects and 1536 objects with hidden attributes from a 64 x 64
    (f, g) grid; object attrs are the subject attrs shifted by (7, 11);
    all 2 x 1536 attribute pairs are globally distinct.
  - Semantic relation: ``R0(a_i) = b_i`` (consistent with the attribute
    shift). Random relation: ``R1(a_i) = b_{pi(i)}``, ``pi`` a random
    permutation of all objects.
  - Split: 1024 probe subjects, 512 held-out subjects (random).

Training
--------
Exactly one epoch (a single pass, shuffled) over a corpus of
  attribute sentences ``x e f_p g_q x`` (each ``attr_repeats`` times)
  + k trained triples ``x a_i r b_t x`` per relation (each ``occurrences``
  times; both relations use the same k subjects).
Plain end-to-end causal LM training with AdamW at a constant learning rate
after a short linear warmup. No component is frozen, embeddings are not
initialized from attributes, and no loss reweighting is used.

Metric (the only one)
---------------------
Ordinary least-squares linear regression (with intercept) from the subject
input embedding to the target object's output embedding (unembedding row).
The regression is always fit on the same fixed probe pairs (all designated
trainable subjects, whether or not their triples were in the corpus), and
evaluated on the held-out subjects: a held-out subject counts as correct
if the predicted vector is closest in cosine distance to the true target
among all object unembedding rows. Chance = 1 / n_objects.

Behavioral recall (argmax next-token prediction on ``x a_i r``) is recorded
as a learning check only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import PROJECT_ROOT, TransformerConfig
from .model import CausalTransformer, build_model
from .synthetic_curriculum import make_disjoint_relation_attrs


@dataclass
class LinearShiftConfig:
    world: str = "category"       # "category" | "onetoone" | "composition"
    subjects_per_bucket: int = 6  # category world size knob (n_subjects = 32 * this)
    comp_nf: int = 64             # composition world size knob (n_subjects = this squared)
    probe: str = "embedding"      # "embedding" | "hidden" (regression input)
    seed: int = 19
    n_triples: int = 128          # k: distinct trained triples per relation
    occurrences: int = 1          # F: occurrences of each triple in the corpus
    attr_repeats: int = 128       # occurrences of each attribute sentence
    object_attr_mode: str = "forward"  # "forward" | "reverse" | "both"
    corpus_order: str = "shuffled"  # "shuffled" | "attrs_first"
    tie_embeddings: bool = False
    n_layers: int = 1
    hidden_dim: int = 64
    n_heads: int = 4
    batch_size: int = 64
    learning_rate: float = 1e-3
    warmup_steps: int = 5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    object_loss_weight: float = 1.0  # optional upweighting of object-token positions
    output_dir: Path = PROJECT_ROOT / "factual_recall" / "results" / "linear_shift"


class World:
    """Entities, attributes, relations, vocabulary, and the probe split."""

    def __init__(self, name: str, seed: int, subjects_per_bucket: int = 6, comp_nf: int = 64) -> None:
        self.name = name
        self.seed = seed
        rng = random.Random(seed)
        if name == "category":
            spb = subjects_per_bucket
            if spb % 3 != 0:
                raise ValueError("subjects_per_bucket must be divisible by 3 (2/3 probe, 1/3 held out)")
            self.n_subjects, self.n_objects = 32 * spb, 32
            self.n_f, self.n_g = 32, spb
            self.subject_attrs = [(i % 32, i // 32) for i in range(self.n_subjects)]
            self.object_attrs = [((j + 3) % 32, 0) for j in range(self.n_objects)]
            self.semantic = {i: i % 32 for i in range(self.n_subjects)}
            # Stratified split: per f-bucket, 2/3 probe + 1/3 held-out subjects.
            n_probe, n_held = 2 * spb // 3, spb // 3
            self.probe_subjects: list[int] = []
            self.heldout_subjects: list[int] = []
            for bucket in range(32):
                members = [bucket + 32 * q for q in range(spb)]
                rng.shuffle(members)
                self.probe_subjects.extend(members[:n_probe])
                self.heldout_subjects.extend(members[n_probe:])
            # Balanced random relation: every object is the target of equally
            # many probe and held-out subjects, independent of attributes.
            self.random: dict[int, int] = {}
            for subjects, per_object in ((self.probe_subjects, n_probe), (self.heldout_subjects, n_held)):
                order = list(subjects)
                rng.shuffle(order)
                targets = [j for j in range(self.n_objects) for _ in range(per_object)]
                rng.shuffle(targets)
                self.random.update(zip(order, targets))
        elif name == "composition":
            # The target is a joint (non-additive) function of both attributes:
            # object index = 8 * class(f) + class(g), with 8 equal-sized attribute
            # classes per attribute. Any function of a single categorical
            # attribute is affine in its one-hot code, so a probe could read it
            # off the static embeddings; a conjunction of two attributes is not,
            # which forces the linear structure (if any) to be computed by the
            # model. Every subject has a unique (f, g) pair, so held-out
            # subjects are unseen attribute combinations.
            nf = comp_nf
            if nf % 8 != 0:
                raise ValueError("comp_nf must be divisible by 8")
            class_size = nf // 8
            self.n_subjects, self.n_objects = nf * nf, 64
            self.n_f = self.n_g = nf
            self.subject_attrs = [(i % nf, i // nf) for i in range(self.n_subjects)]
            self.object_attrs = [((j + 3) % nf, (j + 5) % nf) for j in range(self.n_objects)]
            self.semantic = {
                i: 8 * ((i % nf) // class_size) + ((i // nf) // class_size)
                for i in range(self.n_subjects)
            }
            # Stratified split: per object cell, 3/4 probe + 1/4 held out.
            cell_size = self.n_subjects // self.n_objects
            n_probe, n_held = 3 * cell_size // 4, cell_size // 4
            cells: dict[int, list[int]] = {}
            for i in range(self.n_subjects):
                cells.setdefault(self.semantic[i], []).append(i)
            self.probe_subjects, self.heldout_subjects = [], []
            for cell in sorted(cells):
                members = cells[cell]
                rng.shuffle(members)
                self.probe_subjects.extend(members[:n_probe])
                self.heldout_subjects.extend(members[n_probe:])
            self.random = {}
            for subjects, per_object in ((self.probe_subjects, n_probe), (self.heldout_subjects, n_held)):
                order = list(subjects)
                rng.shuffle(order)
                targets = [j for j in range(self.n_objects) for _ in range(per_object)]
                rng.shuffle(targets)
                self.random.update(zip(order, targets))
        elif name == "onetoone":
            self.n_subjects = self.n_objects = 1536
            self.n_f, self.n_g = 64, 64
            subject_attrs, object_attrs = make_disjoint_relation_attrs(
                n_entities=self.n_subjects, n_f=64, n_g=64, delta_f=7, delta_g=11, seed=seed
            )
            self.subject_attrs = list(subject_attrs)
            self.object_attrs = list(object_attrs)
            self.semantic = {i: i for i in range(self.n_subjects)}
            permutation = list(range(self.n_objects))
            rng.shuffle(permutation)
            self.random = dict(enumerate(permutation))
            subjects = list(range(self.n_subjects))
            rng.shuffle(subjects)
            self.probe_subjects = sorted(subjects[:1024])
            self.heldout_subjects = sorted(subjects[1024 : 1024 + 512])
        else:
            raise ValueError(name)
        self.probe_subjects.sort()
        self.heldout_subjects.sort()

        # Token layout: x, r_0, r_1, subjects, objects, f attrs, g attrs.
        self.sep, self.r0, self.r1 = 0, 1, 2
        self.subject_offset = 3
        self.object_offset = self.subject_offset + self.n_subjects
        self.f_offset = self.object_offset + self.n_objects
        self.g_offset = self.f_offset + self.n_f
        self.vocab_size = self.g_offset + self.n_g

    def subject(self, i: int) -> int:
        return self.subject_offset + i

    def object(self, j: int) -> int:
        return self.object_offset + j

    def target(self, relation: str, i: int) -> int:
        return {"semantic": self.semantic, "random": self.random}[relation][i]

    def relation_token(self, relation: str) -> int:
        return self.r0 if relation == "semantic" else self.r1


def build_corpus(cfg: LinearShiftConfig, world: World) -> tuple[list[list[int]], list[int]]:
    """Return (sequences, trained subjects). One corpus = one epoch."""
    rng = random.Random(cfg.seed + 1)
    attrs: list[list[int]] = []
    for i in range(world.n_subjects):
        f, g = world.subject_attrs[i]
        attrs.append([world.sep, world.subject(i), world.f_offset + f, world.g_offset + g, world.sep])
    for j in range(world.n_objects):
        f, g = world.object_attrs[j]
        if cfg.object_attr_mode in ("forward", "both"):
            attrs.append([world.sep, world.object(j), world.f_offset + f, world.g_offset + g, world.sep])
        if cfg.object_attr_mode in ("reverse", "both"):
            attrs.append([world.sep, world.f_offset + f, world.g_offset + g, world.object(j), world.sep])

    trained = list(world.probe_subjects)
    rng.shuffle(trained)
    trained = sorted(trained[: cfg.n_triples])
    triples: list[list[int]] = []
    for i in trained:
        for relation in ("semantic", "random"):
            seq = [
                world.sep,
                world.subject(i),
                world.relation_token(relation),
                world.object(world.target(relation, i)),
                world.sep,
            ]
            triples.extend([seq] * cfg.occurrences)

    attr_sequences = attrs * cfg.attr_repeats
    if cfg.corpus_order == "shuffled":
        sequences = attr_sequences + triples
        rng.shuffle(sequences)
    elif cfg.corpus_order == "attrs_first":
        rng.shuffle(attr_sequences)
        rng.shuffle(triples)
        sequences = attr_sequences + triples
    else:
        raise ValueError(cfg.corpus_order)
    return sequences, trained


def train_one_epoch(
    cfg: LinearShiftConfig, world: World, sequences: list[list[int]], device: torch.device
) -> tuple[CausalTransformer, int, float]:
    torch.manual_seed(cfg.seed)
    model_cfg = TransformerConfig(
        n_layers=cfg.n_layers,
        hidden_dim=cfg.hidden_dim,
        n_heads=cfg.n_heads,
        context_len=8,
        tie_embeddings=cfg.tie_embeddings,
    )
    model = build_model(model_cfg, world)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Batches follow corpus order; build_corpus already shuffled the corpus
    # (fully, or within the attribute and triple groups for "attrs_first").
    data = torch.tensor(sequences, dtype=torch.long)
    n_steps = (len(data) + cfg.batch_size - 1) // cfg.batch_size
    last_loss = float("nan")
    for step in range(n_steps):
        batch = data[step * cfg.batch_size : (step + 1) * cfg.batch_size].to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        warmup = min(1.0, (step + 1) / max(1, cfg.warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate * warmup
        logits = model(x)
        flat_logits = logits.reshape(-1, world.vocab_size)
        flat_y = y.reshape(-1)
        if cfg.object_loss_weight != 1.0:
            losses = nn.functional.cross_entropy(flat_logits, flat_y, reduction="none")
            is_object = (flat_y >= world.object_offset) & (flat_y < world.f_offset)
            weights = torch.where(is_object, cfg.object_loss_weight, 1.0)
            loss = (losses * weights).sum() / weights.sum()
        else:
            loss = nn.functional.cross_entropy(flat_logits, flat_y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        last_loss = float(loss.item())
    return model, n_steps, last_loss


def prompt_hidden_states(
    model: CausalTransformer, world: World, relation: str, subjects: list[int], device: torch.device
) -> np.ndarray:
    """Final-layer hidden state (after ln_f) at the last position of ``x a_i r``."""
    relation_token = world.relation_token(relation)
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(subjects), 512):
            chunk = subjects[start : start + 512]
            x = torch.tensor([[world.sep, world.subject(i), relation_token] for i in chunk], device=device)
            with model.capture_residuals() as residuals:
                model(x)
            hidden = model.ln_f(residuals[-1])[:, -1, :]
            rows.append(hidden.detach().cpu().numpy().astype(np.float64))
    model.train()
    return np.concatenate(rows, axis=0)


def ols_probe(
    model: CausalTransformer, world: World, relation: str, mode: str, device: torch.device
) -> dict[str, float]:
    """The metric: OLS affine map from the subject representation (static
    input embedding, or final hidden state of the prompt ``x a_i r``) to the
    target object's unembedding row."""
    from sklearn.linear_model import LinearRegression

    emb = model.token_emb.weight.detach().cpu().numpy().astype(np.float64)
    unemb = model.lm_head.weight.detach().cpu().numpy().astype(np.float64)
    object_rows = unemb[[world.object(j) for j in range(world.n_objects)]]

    if mode == "embedding":
        fit_x = emb[[world.subject(i) for i in world.probe_subjects]]
        eval_x = emb[[world.subject(i) for i in world.heldout_subjects]]
    elif mode == "hidden":
        fit_x = prompt_hidden_states(model, world, relation, world.probe_subjects, device)
        eval_x = prompt_hidden_states(model, world, relation, world.heldout_subjects, device)
    else:
        raise ValueError(mode)
    fit_y = object_rows[[world.target(relation, i) for i in world.probe_subjects]]
    eval_targets = np.array([world.target(relation, i) for i in world.heldout_subjects])

    reg = LinearRegression(fit_intercept=True).fit(fit_x, fit_y)
    pred = reg.predict(eval_x)
    eval_y = object_rows[eval_targets]
    pred_norm = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    cand_norm = object_rows / np.maximum(np.linalg.norm(object_rows, axis=1, keepdims=True), 1e-12)
    similarities = pred_norm @ cand_norm.T
    accuracy = float(np.mean(similarities.argmax(axis=1) == eval_targets))

    ss_res = float(np.square(eval_y - pred).sum())
    ss_tot = float(np.square(eval_y - eval_y.mean(axis=0, keepdims=True)).sum())
    denom = np.maximum(np.linalg.norm(pred, axis=1) * np.linalg.norm(eval_y, axis=1), 1e-12)
    return {
        "probe_accuracy": accuracy,
        "probe_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "probe_cosine": float(np.mean(np.sum(pred * eval_y, axis=1) / denom)),
    }


def behavioral_check(
    model: CausalTransformer, world: World, relation: str, subjects: list[int], device: torch.device
) -> dict[str, float] | None:
    """Learning check (not an analysis metric): next-token recall and loss."""
    if not subjects:
        return None
    relation_token = world.relation_token(relation)
    model.eval()
    recall_n, loss_sum = 0, 0.0
    with torch.no_grad():
        for start in range(0, len(subjects), 256):
            chunk = subjects[start : start + 256]
            x = torch.tensor([[world.sep, world.subject(i), relation_token] for i in chunk], device=device)
            y = torch.tensor([world.object(world.target(relation, i)) for i in chunk], device=device)
            logits = model(x)[:, -1, :]
            recall_n += int((logits.argmax(dim=-1) == y).sum().item())
            loss_sum += float(nn.functional.cross_entropy(logits, y, reduction="sum").item())
    model.train()
    return {"recall": recall_n / len(subjects), "loss": loss_sum / len(subjects)}


def run_single(cfg: LinearShiftConfig, device: torch.device) -> dict:
    world = World(cfg.world, cfg.seed, subjects_per_bucket=cfg.subjects_per_bucket, comp_nf=cfg.comp_nf)
    sequences, trained_subjects = build_corpus(cfg, world)
    t0 = time.time()
    model, n_steps, last_loss = train_one_epoch(cfg, world, sequences, device)

    result: dict = {
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "epochs": 1,
        "corpus_sequences": len(sequences),
        "train_steps": n_steps,
        "last_train_loss": last_loss,
        "num_parameters": model.n_parameters(),
        "trained_triples_per_relation": len(trained_subjects),
        "probe_fit_pairs": len(world.probe_subjects),
        "heldout_subjects": len(world.heldout_subjects),
        "elapsed_sec": time.time() - t0,
        "relations": {},
    }
    for relation in ("semantic", "random"):
        entry = ols_probe(model, world, relation, cfg.probe, device)
        entry["trained_check"] = behavioral_check(model, world, relation, trained_subjects, device)
        entry["heldout_check"] = behavioral_check(model, world, relation, world.heldout_subjects, device)
        result["relations"][relation] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=str, default="category")
    parser.add_argument("--subjects-per-bucket", type=int, default=6)
    parser.add_argument("--probe", type=str, default="embedding")
    parser.add_argument("--comp-nf", type=int, default=64)
    parser.add_argument("--object-loss-weight", type=float, default=1.0)
    parser.add_argument("--seeds", type=str, default="19")
    parser.add_argument("--k-values", type=str, default="128", help="distinct trained triples per relation")
    parser.add_argument("--occurrences", type=str, default="1", help="occurrences per triple in the corpus")
    parser.add_argument("--attr-repeats", type=int, default=128)
    parser.add_argument("--corpus-order", type=str, default="shuffled")
    parser.add_argument("--object-attr-mode", type=str, default="forward")
    parser.add_argument("--tie-embeddings", action="store_true")
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="factual_recall/results/linear_shift")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        for k in [int(s) for s in args.k_values.split(",")]:
            for occ in [int(s) for s in args.occurrences.split(",")]:
                cfg = LinearShiftConfig(
                    world=args.world,
                    subjects_per_bucket=args.subjects_per_bucket,
                    comp_nf=args.comp_nf,
                    probe=args.probe,
                    seed=seed,
                    n_triples=k,
                    occurrences=occ,
                    attr_repeats=args.attr_repeats,
                    corpus_order=args.corpus_order,
                    object_attr_mode=args.object_attr_mode,
                    tie_embeddings=args.tie_embeddings,
                    n_layers=args.n_layers,
                    hidden_dim=args.hidden_dim,
                    learning_rate=args.learning_rate,
                    warmup_steps=args.warmup_steps,
                    object_loss_weight=args.object_loss_weight,
                    output_dir=output_dir,
                )
                result = run_single(cfg, device)
                runs.append(result)
                sem = result["relations"]["semantic"]
                rnd = result["relations"]["random"]
                sem_recall = (sem["trained_check"] or {}).get("recall", float("nan"))
                rnd_recall = (rnd["trained_check"] or {}).get("recall", float("nan"))
                print(
                    f"world={args.world} seed={seed} k={k:5d} occ={occ:3d} | "
                    f"probe sem={sem['probe_accuracy']:.3f} rnd={rnd['probe_accuracy']:.3f} | "
                    f"train recall sem={sem_recall:.3f} rnd={rnd_recall:.3f}",
                    flush=True,
                )
    out_path = output_dir / "linear_shift_metrics.json"
    out_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
