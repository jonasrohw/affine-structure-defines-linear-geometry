"""Experiment 1: one epoch on matched attribute-defined and random relations."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import PROJECT_ROOT, TransformerConfig
from .model import CausalTransformer, build_model


@dataclass
class LinearShiftConfig:
    world: str = "category"
    subjects_per_bucket: int = 6  # category world size knob (n_subjects = 32 * this)
    probe: str = "embedding"
    seed: int = 19
    n_triples: int = 128          # k: distinct trained triples per relation
    occurrences: int = 1          # F: occurrences of each triple in the corpus
    attr_repeats: int = 128       # occurrences of each attribute sentence
    tie_embeddings: bool = False
    n_layers: int = 1
    hidden_dim: int = 64
    n_heads: int = 4
    batch_size: int = 64
    learning_rate: float = 1e-3
    warmup_steps: int = 5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    output_dir: Path = PROJECT_ROOT / "factual_recall" / "results" / "linear_shift"


class World:
    """Entities, attributes, relations, vocabulary, and the probe split."""

    def __init__(self, name: str, seed: int, subjects_per_bucket: int = 6) -> None:
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
        attrs.append([world.sep, world.object(j), world.f_offset + f, world.g_offset + g, world.sep])

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
    sequences = attr_sequences + triples
    rng.shuffle(sequences)
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
    # as one shuffled epoch.
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
        loss = nn.functional.cross_entropy(flat_logits, flat_y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        last_loss = float(loss.item())
    return model, n_steps, last_loss


def ols_probe(
    model: CausalTransformer, world: World, relation: str, mode: str, device: torch.device
) -> dict[str, float]:
    """The metric: OLS affine map from the subject representation (static
    input embedding) to the
    target object's unembedding row."""
    from sklearn.linear_model import LinearRegression

    emb = model.token_emb.weight.detach().cpu().numpy().astype(np.float64)
    unemb = model.lm_head.weight.detach().cpu().numpy().astype(np.float64)
    object_rows = unemb[[world.object(j) for j in range(world.n_objects)]]

    if mode != "embedding":
        raise ValueError("The paper probes raw input embeddings")
    fit_x = emb[[world.subject(i) for i in world.probe_subjects]]
    eval_x = emb[[world.subject(i) for i in world.heldout_subjects]]
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
    world = World(cfg.world, cfg.seed, subjects_per_bucket=cfg.subjects_per_bucket)
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
