"""
Training loop for the Graz experiment.
"""
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .data_generation import token_type_mask
from .model import MiniTransformer


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float) -> float:
    if step < warmup_steps:
        return lr * step / warmup_steps
    t = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return lr * 0.5 * (1.0 + math.cos(math.pi * t))


def build_valid_attr_table(cfg: Config, device: torch.device) -> torch.Tensor:
    """Boolean tensor (vocab_size, vocab_size): valid[entity, attr] = True."""
    vocab = cfg.vocab
    valid = torch.zeros(vocab.vocab_size, vocab.vocab_size, dtype=torch.bool, device=device)
    for i in range(vocab.n_entities):
        for token in [vocab.subject_id(i), vocab.object_id(i)]:
            for attr_id in vocab.valid_attr_ids(token):
                valid[token, attr_id] = True
    return valid


def evaluate(
    model: MiniTransformer,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    valid_table: torch.Tensor,
    max_batches: int = 20,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    attr_loss = 0.0
    attr_valid_correct = 0
    attr_total = 0
    n_batches = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B * T, V), y.view(B * T))
            total_loss += loss.item()

            _, entity_mask, a_mask, _ = token_type_mask(y, cfg.vocab)

            x_entity = (x >= cfg.vocab.subject_offset) & (x < cfg.vocab.f_attr_offset)
            y_attr = (y >= cfg.vocab.f_attr_offset) & (y < cfg.vocab.relation_offset)
            attr_pos = x_entity & y_attr

            if attr_pos.sum() > 0:
                attr_logits = logits[attr_pos]
                attr_targets = y[attr_pos]
                attr_loss += nn.functional.cross_entropy(attr_logits, attr_targets).item()

                preds = attr_logits.argmax(dim=-1)
                entities = x[attr_pos]
                attr_valid_correct += valid_table[entities, preds].sum().item()
                attr_total += attr_pos.sum().item()

            n_batches += 1

    metrics = {
        "loss": total_loss / max(1, n_batches),
        "loss_attr": attr_loss / max(1, n_batches),
        "acc_attr_valid": attr_valid_correct / max(1, attr_total),
    }
    model.train()
    return metrics


def train(
    model: MiniTransformer,
    train_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    val_loader: Optional[DataLoader] = None,
    verbose: bool = True,
) -> Dict[str, list]:
    model.to(device)
    model.train()

    valid_table = build_valid_attr_table(cfg, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * cfg.train.max_epochs

    history = {"train_loss": [], "val_loss": [], "val_acc_attr": []}
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

            epoch_loss += loss.item()
            global_step += 1

        avg_train_loss = epoch_loss / steps_per_epoch
        history["train_loss"].append(avg_train_loss)

        if (epoch + 1) % cfg.train.eval_every == 0 or epoch == cfg.train.max_epochs - 1:
            val_metrics = {}
            if val_loader is not None:
                val_metrics = evaluate(model, val_loader, device, cfg, valid_table)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc_attr"].append(val_metrics["acc_attr_valid"])

                if val_metrics["acc_attr_valid"] > best_val_acc:
                    best_val_acc = val_metrics["acc_attr_valid"]
                    torch.save({
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_acc_attr": best_val_acc,
                        "cfg": cfg,
                    }, best_ckpt)

            if verbose:
                elapsed = time.time() - t0
                msg = (f"Epoch {epoch+1:4d}/{cfg.train.max_epochs} | "
                       f"train_loss={avg_train_loss:.4f} | lr={lr:.2e} | {elapsed:.0f}s")
                if val_metrics:
                    msg += (f" | loss_attr={val_metrics['loss_attr']:.4f}"
                            f" | attr_valid={val_metrics['acc_attr_valid']:.3f}")
                print(msg)

    return history


def evaluate_relations(
    model: MiniTransformer,
    cfg: Config,
    device: torch.device,
    subject_indices: list,
) -> dict:
    """For each subject and relation, present [x, r_k, a_i] and check object prediction."""
    model.eval()
    vocab = cfg.vocab
    correct = {k: 0 for k in range(vocab.num_relations)}
    total = {k: 0 for k in range(vocab.num_relations)}

    with torch.no_grad():
        for subj_idx in subject_indices:
            for rel_k in range(vocab.num_relations):
                context = torch.tensor(
                    [[vocab.sep_id, vocab.relation_id(rel_k), vocab.subject_id(subj_idx)]],
                    dtype=torch.long, device=device,
                )
                logits = model(context)
                pred = logits[0, -1].argmax().item()
                target = vocab.object_id(vocab.relation_object(subj_idx, rel_k))
                correct[rel_k] += int(pred == target)
                total[rel_k] += 1

    results = {}
    for k in range(vocab.num_relations):
        results[f"r{k}_acc"] = correct[k] / max(1, total[k])
    results["overall_acc"] = sum(correct.values()) / max(1, sum(total.values()))
    return results


def load_checkpoint(path: Path, model: MiniTransformer, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt
