"""
Top-k coverage analysis for attribute and entity prediction.

Two metrics:
1. Attribute precision@2: for each entity, are both valid f-attrs (g-attrs)
   in the top 2 of the 10 f-attr (g-attr) logits?

2. Entity precision@405: for each attribute, in the reverse context
   [sep, marker, attr], are all 405 valid entities in the top 405
   entity predictions? (9 types × 45 entities/type = 405)

Usage:
    python experiments/analysis_topk.py --checkpoint checkpoints_v3_seed42/final.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.config import Config, ModelConfig, TrainConfig, VocabConfig
from src.model import build_model


@torch.no_grad()
def eval_attr_precision_at_k(model, vocab, device, batch_size=512):
    """For each entity, are all k valid f-attrs in the top k? Same for g-attrs."""
    model.eval()
    k = len(vocab.combos[0])  # k_per_group (2 for C(10,2))

    entities = []
    for i in range(vocab.n_entities):
        entities.append(('subject', i, vocab.subject_id(i)))
    for j in range(vocab.n_entities):
        entities.append(('object', j, vocab.object_id(j)))

    f_correct = 0
    g_correct = 0
    total = 0

    for start in range(0, len(entities), batch_size):
        batch = entities[start:start + batch_size]
        contexts = torch.tensor(
            [[vocab.sep_id, tok] for _, _, tok in batch],
            dtype=torch.long, device=device,
        )
        logits = model(contexts)[:, -1]  # (B, vocab_size)

        f_logits = logits[:, vocab.f_attr_offset:vocab.f_attr_offset + vocab.n_f_attrs]
        g_logits = logits[:, vocab.g_attr_offset:vocab.g_attr_offset + vocab.n_g_attrs]

        f_topk = f_logits.topk(k, dim=-1).indices  # (B, k)
        g_topk = g_logits.topk(k, dim=-1).indices

        for i, (role, idx, _) in enumerate(batch):
            if role == 'subject':
                valid_f = set(vocab.subject_f_type(idx))
                valid_g = set(vocab.subject_g_type(idx))
            else:
                valid_f = set(vocab.object_f_type(idx))
                valid_g = set(vocab.object_g_type(idx))

            f_correct += len(set(f_topk[i].tolist()) & valid_f)
            g_correct += len(set(g_topk[i].tolist()) & valid_g)
            total += 1

    return {
        'f_precision_at_k': f_correct / (total * k),
        'g_precision_at_k': g_correct / (total * k),
        'overall': (f_correct + g_correct) / (total * 2 * k),
        'k': k,
    }


@torch.no_grad()
def eval_entity_precision_at_k(model, vocab, device):
    """For each attribute, are all 405 valid entities in the top 405 predictions?

    Uses reverse-short context: [sep, S/O_marker, attr] → predict entity.
    """
    model.eval()

    n_types = vocab.n_types
    entities_per_type = vocab.n_entities // n_types
    n_attrs_per_type = len(vocab.combos[0])  # k_per_group = 2
    # Each attr appears in C(n_f_attrs-1, k-1) types = C(9,1) = 9
    from math import comb
    types_per_attr = comb(vocab.n_f_attrs - 1, n_attrs_per_type - 1)
    k_entity = types_per_attr * entities_per_type  # 9 × 45 = 405
    chance = k_entity / vocab.n_entities

    # Verify k_entity
    test_count = sum(1 for i in range(vocab.n_entities) if 0 in vocab.subject_f_type(i))
    assert test_count == k_entity, f"Expected {k_entity}, got {test_count}"

    # Precompute valid entity sets per (role, attr_group, attr_value)
    valid_sets = {}
    for a in range(vocab.n_f_attrs):
        valid_sets[('subject', 'f', a)] = {
            i for i in range(vocab.n_entities) if a in vocab.subject_f_type(i)
        }
        valid_sets[('object', 'f', a)] = {
            j for j in range(vocab.n_entities) if a in vocab.object_f_type(j)
        }
    for a in range(vocab.n_g_attrs):
        valid_sets[('subject', 'g', a)] = {
            i for i in range(vocab.n_entities) if a in vocab.subject_g_type(i)
        }
        valid_sets[('object', 'g', a)] = {
            j for j in range(vocab.n_entities) if a in vocab.object_g_type(j)
        }

    results = {'subject': [], 'object': []}

    for role in ['subject', 'object']:
        marker = vocab.s_marker_id if role == 'subject' else vocab.o_marker_id
        offset = vocab.subject_offset if role == 'subject' else vocab.object_offset

        # Build all contexts: [sep, marker, attr] for each attribute
        attrs = []
        for a in range(vocab.n_f_attrs):
            attrs.append(('f', a, vocab.f_attr_id(a)))
        for a in range(vocab.n_g_attrs):
            attrs.append(('g', a, vocab.g_attr_id(a)))

        contexts = torch.tensor(
            [[vocab.sep_id, marker, attr_token] for _, _, attr_token in attrs],
            dtype=torch.long, device=device,
        )
        logits = model(contexts)[:, -1]  # (n_attrs, vocab_size)
        entity_logits = logits[:, offset:offset + vocab.n_entities]  # (n_attrs, n_entities)

        topk_all = entity_logits.topk(k_entity, dim=-1).indices  # (n_attrs, k_entity)

        for i, (group, a, _) in enumerate(attrs):
            valid = valid_sets[(role, group, a)]
            topk = set(topk_all[i].tolist())
            precision = len(topk & valid) / k_entity
            results[role].append(precision)

    subj_mean = np.mean(results['subject'])
    obj_mean = np.mean(results['object'])
    return {
        'subject_precision_at_k': subj_mean,
        'object_precision_at_k': obj_mean,
        'overall': (subj_mean + obj_mean) / 2,
        'k': k_entity,
        'chance': chance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(__file__).parent.parent / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    vocab_kwargs = ckpt["vocab_kwargs"]
    held_out = ckpt.get("held_out_triples", [])
    vocab = VocabConfig(**vocab_kwargs, held_out_triples=held_out)

    mcfg = ckpt["model_cfg"]
    model_cfg = ModelConfig(n_layers=mcfg["n_layers"], hidden_dim=mcfg["hidden_dim"],
                            n_heads=mcfg["n_heads"])
    model = build_model(model_cfg, vocab)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Entities: {vocab.n_entities}, Types: {vocab.n_types}, "
          f"k_per_group: {len(vocab.combos[0])}")
    print()

    # 1. Attribute precision@k
    print("=" * 60)
    print("Attribute precision@k (entity → attribute prediction)")
    print("=" * 60)
    attr_res = eval_attr_precision_at_k(model, vocab, device)
    print(f"  k = {attr_res['k']} (k_per_group)")
    print(f"  f-attr precision@{attr_res['k']}: {attr_res['f_precision_at_k']:.4f}")
    print(f"  g-attr precision@{attr_res['k']}: {attr_res['g_precision_at_k']:.4f}")
    print(f"  overall:                          {attr_res['overall']:.4f}")
    print(f"  chance: {1/vocab.n_f_attrs:.4f}")
    print()

    # 2. Entity precision@k
    print("=" * 60)
    print("Entity precision@k (attribute → entity prediction)")
    print("=" * 60)
    ent_res = eval_entity_precision_at_k(model, vocab, device)
    print(f"  k = {ent_res['k']} (types_per_attr × entities_per_type)")
    print(f"  subject precision@{ent_res['k']}: {ent_res['subject_precision_at_k']:.4f}")
    print(f"  object  precision@{ent_res['k']}: {ent_res['object_precision_at_k']:.4f}")
    print(f"  overall:                          {ent_res['overall']:.4f}")
    print(f"  chance: {ent_res['chance']:.4f}")


if __name__ == "__main__":
    main()
