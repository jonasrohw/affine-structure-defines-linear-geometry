"""Training and evaluation loops for factual recall models."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .datasets import token_type_mask
from .model import CausalTransformer


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float) -> float:
    if step < warmup_steps:
        return lr * step / warmup_steps
    t = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return lr * 0.5 * (1.0 + math.cos(math.pi * t))


def evaluate(
    model: CausalTransformer,
    loader: DataLoader,
    device: torch.device,
    cfg: ExperimentConfig,
    max_batches: int = 20,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = {"sep": 0, "subj": 0, "rel": 0, "filler": 0, "obj": 0}
    total = {"sep": 0, "subj": 0, "rel": 0, "filler": 0, "obj": 0}

    with torch.no_grad():
        seen_batches = 0
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            seen_batches += 1
            x, y = x.to(device), y.to(device)
            logits = model(x)
            batch_size, seq_len, vocab_size = logits.shape
            loss = nn.functional.cross_entropy(
                logits.view(batch_size * seq_len, vocab_size),
                y.view(batch_size * seq_len),
            )
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            sep_m, subj_m, rel_m, filler_m, obj_m = token_type_mask(y, cfg.vocab)
            for key, mask in (
                ("sep", sep_m),
                ("subj", subj_m),
                ("rel", rel_m),
                ("filler", filler_m),
                ("obj", obj_m),
            ):
                correct[key] += (preds[mask] == y[mask]).sum().item()
                total[key] += mask.sum().item()

    metrics = {"loss": total_loss / max(1, seen_batches)}
    for key in ("sep", "subj", "rel", "filler", "obj"):
        metrics[f"acc_{key}"] = correct[key] / max(1, total[key])
    model.train()
    return metrics


def train(
    model: CausalTransformer,
    train_loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
    val_loader: Optional[DataLoader] = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * cfg.train.max_epochs
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc_obj": []}
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
            for group in optimizer.param_groups:
                group["lr"] = lr

            logits = model(x)
            batch_size, seq_len, vocab_size = logits.shape
            loss = nn.functional.cross_entropy(
                logits.view(batch_size * seq_len, vocab_size),
                y.view(batch_size * seq_len),
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1

        avg_train_loss = epoch_loss / max(1, steps_per_epoch)
        history["train_loss"].append(avg_train_loss)

        if (epoch + 1) % cfg.train.eval_every == 0 or epoch == cfg.train.max_epochs - 1:
            val_metrics = {}
            if val_loader is not None:
                val_metrics = evaluate(model, val_loader, device, cfg)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc_obj"].append(val_metrics["acc_obj"])
                if val_metrics["acc_obj"] > best_val_acc:
                    best_val_acc = val_metrics["acc_obj"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "val_acc_obj": best_val_acc,
                            "cfg": cfg,
                        },
                        best_ckpt,
                    )

            if verbose:
                elapsed = time.time() - t0
                msg = (
                    f"Epoch {epoch + 1:4d}/{cfg.train.max_epochs} | "
                    f"train_loss={avg_train_loss:.4f} | lr={lr:.2e} | {elapsed:.0f}s"
                )
                if val_metrics:
                    msg += (
                        f" | val_loss={val_metrics['loss']:.4f}"
                        f" | acc_obj={val_metrics['acc_obj']:.3f}"
                        f" | acc_rel={val_metrics['acc_rel']:.3f}"
                        f" | acc_filler={val_metrics['acc_filler']:.3f}"
                    )
                print(msg)

    return history


def load_checkpoint(path: Path, model: CausalTransformer, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint
