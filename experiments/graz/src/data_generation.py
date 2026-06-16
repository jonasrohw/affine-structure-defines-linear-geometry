"""
Data generation for the Graz experiment.

Phase 1 sequences: concatenations of 3-token attribute sentences:
    x  entity  attribute    x  entity  attribute    ...

Phase 2 sequences: mix of 3-token attribute sentences and 4-token relation triples:
    x  entity  attribute    x  r_k  subject  object    ...

Each attribute sentence associates an entity with one of its attributes.
Each relation triple states that subject i is related to object j via r_k.
"""
import random
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config, VocabConfig

ATTR_SENTENCE_LEN = 3
REL_SENTENCE_LEN = 4
REVERSE_SENTENCE_LEN = 8
REVERSE_SHORT_SENTENCE_LEN = 4
CONTRASTIVE_SENTENCE_LEN = 5


def generate_attr_sentence(entity_token: int, vocab: VocabConfig, rng: random.Random) -> List[int]:
    f_attrs, g_attrs = vocab.entity_attrs(entity_token)
    if rng.random() < 0.5:
        attr = rng.choice(f_attrs)
        attr_token = vocab.f_attr_id(attr)
    else:
        attr = rng.choice(g_attrs)
        attr_token = vocab.g_attr_id(attr)
    return [vocab.sep_id, entity_token, attr_token]


def generate_reverse_sentence(vocab: VocabConfig, rng: random.Random) -> List[int]:
    """Generate [x, f1, f2, f3, g1, g2, g3, entity] — all 6 attrs → entity.

    Unambiguous: the full (f-type, g-type) profile uniquely identifies every
    subject and object.
    """
    if rng.random() < 0.5:
        idx = rng.randint(0, vocab.n_entities - 1)
        entity_token = vocab.object_id(idx)
        f_attrs = list(vocab.object_f_type(idx))
        g_attrs = list(vocab.object_g_type(idx))
    else:
        idx = rng.randint(0, vocab.n_entities - 1)
        entity_token = vocab.subject_id(idx)
        f_attrs = list(vocab.subject_f_type(idx))
        g_attrs = list(vocab.subject_g_type(idx))
    f_tokens = [vocab.f_attr_id(a) for a in f_attrs]
    g_tokens = [vocab.g_attr_id(a) for a in g_attrs]
    rng.shuffle(f_tokens)
    rng.shuffle(g_tokens)
    return [vocab.sep_id] + f_tokens + g_tokens + [entity_token]


def generate_reverse_short_sentence(vocab: VocabConfig, rng: random.Random) -> List[int]:
    """Generate [x, S/O, attr, entity] — one attribute + role marker → entity.

    Randomly picks a subject or object, then picks one of its 6 attributes.
    Prefixes with S (subject) or O (object) marker so the model knows the role.
    """
    if rng.random() < 0.5:
        idx = rng.randint(0, vocab.n_entities - 1)
        entity_token = vocab.subject_id(idx)
        marker = vocab.s_marker_id
        f_attrs = list(vocab.subject_f_type(idx))
        g_attrs = list(vocab.subject_g_type(idx))
    else:
        idx = rng.randint(0, vocab.n_entities - 1)
        entity_token = vocab.object_id(idx)
        marker = vocab.o_marker_id
        f_attrs = list(vocab.object_f_type(idx))
        g_attrs = list(vocab.object_g_type(idx))
    if rng.random() < 0.5:
        attr_token = vocab.f_attr_id(rng.choice(f_attrs))
    else:
        attr_token = vocab.g_attr_id(rng.choice(g_attrs))
    return [vocab.sep_id, marker, attr_token, entity_token]


def generate_contrastive_sentence(vocab: VocabConfig, rng: random.Random,
                                  train_triples: Optional[List] = None) -> List[int]:
    """Generate [x, Qk, object, subject, y/n] — positive or negative relation example.

    Uses dedicated query markers Q0/Q1 and reversed order (obj before subj)
    to clearly differentiate from generative relation sentences [x, r_k, subj, obj].
    """
    if train_triples is not None:
        subj_idx, rel_k = rng.choice(train_triples)
    else:
        subj_idx = rng.choice(vocab.train_subject_indices)
        rel_k = rng.randint(0, vocab.num_relations - 1)
    valid_objects = set(vocab.relation_objects(subj_idx, rel_k))
    if rng.random() < 0.5:
        obj_idx = rng.choice(list(valid_objects))
        label = vocab.y_id
    else:
        invalid = [j for j in range(vocab.n_entities) if j not in valid_objects]
        obj_idx = rng.choice(invalid)
        label = vocab.n_id
    return [vocab.sep_id, vocab.query_id(rel_k), vocab.object_id(obj_idx),
            vocab.subject_id(subj_idx), label]


def generate_rel_sentence(vocab: VocabConfig, rng: random.Random,
                          train_triples: Optional[List] = None) -> List[int]:
    if train_triples is not None:
        subj_idx, rel_k = rng.choice(train_triples)
    else:
        subj_idx = rng.choice(vocab.train_subject_indices)
        rel_k = rng.randint(0, vocab.num_relations - 1)
    objects = vocab.relation_objects(subj_idx, rel_k)
    obj_idx = objects[0] if len(objects) == 1 else rng.choice(objects)
    return [vocab.sep_id, vocab.relation_id(rel_k), vocab.subject_id(subj_idx), vocab.object_id(obj_idx)]


def generate_sequence(length: int, vocab: VocabConfig, rng: random.Random) -> List[int]:
    all_entities = ([vocab.subject_id(i) for i in range(vocab.n_entities)] +
                    [vocab.object_id(j) for j in range(vocab.n_entities)])
    tokens: List[int] = []
    while len(tokens) + ATTR_SENTENCE_LEN <= length:
        entity = rng.choice(all_entities)
        tokens.extend(generate_attr_sentence(entity, vocab, rng))
    return tokens


def generate_mixed_sequence(length: int, vocab: VocabConfig, rng: random.Random,
                            p_relation: float = 0.25,
                            train_triples: Optional[List] = None,
                            p_reverse: float = 0.0,
                            p_reverse_short: float = 0.0,
                            p_contrastive: float = 0.0) -> List[int]:
    all_entities = ([vocab.subject_id(i) for i in range(vocab.n_entities)] +
                    [vocab.object_id(j) for j in range(vocab.n_entities)])
    tokens: List[int] = []
    while True:
        remaining = length - len(tokens)
        if remaining < ATTR_SENTENCE_LEN:
            break
        r = rng.random()
        if r < p_relation and remaining >= REL_SENTENCE_LEN:
            tokens.extend(generate_rel_sentence(vocab, rng, train_triples))
        elif r < p_relation + p_contrastive and remaining >= CONTRASTIVE_SENTENCE_LEN:
            tokens.extend(generate_contrastive_sentence(vocab, rng, train_triples))
        elif r < p_relation + p_contrastive + p_reverse and remaining >= REVERSE_SENTENCE_LEN:
            tokens.extend(generate_reverse_sentence(vocab, rng))
        elif r < p_relation + p_contrastive + p_reverse + p_reverse_short and remaining >= REVERSE_SHORT_SENTENCE_LEN:
            tokens.extend(generate_reverse_short_sentence(vocab, rng))
        else:
            entity = rng.choice(all_entities)
            tokens.extend(generate_attr_sentence(entity, vocab, rng))
    while len(tokens) < length:
        tokens.append(vocab.sep_id)
    return tokens


def generate_exhaustive_contrastive(vocab: VocabConfig, train_triples: List,
                                    seq_length: int, seed: int = 0) -> List[List[int]]:
    """Generate all positive contrastive sentences exhaustively as a flat list.

    For each training (subject, relation) pair, generates one positive example
    per valid object and one matched negative (random invalid object).
    Returns a list of 5-token sentences (not packed into sequences).
    """
    rng = random.Random(seed)
    sentences = []
    for subj_idx, rel_k in train_triples:
        valid_objects = list(vocab.relation_objects(subj_idx, rel_k))
        invalid_objects = [j for j in range(vocab.n_entities) if j not in set(valid_objects)]
        for obj_idx in valid_objects:
            sentences.append([vocab.sep_id, vocab.query_id(rel_k),
                              vocab.object_id(obj_idx), vocab.subject_id(subj_idx), vocab.y_id])
        for _ in range(len(valid_objects)):
            obj_idx = rng.choice(invalid_objects)
            sentences.append([vocab.sep_id, vocab.query_id(rel_k),
                              vocab.object_id(obj_idx), vocab.subject_id(subj_idx), vocab.n_id])
    rng.shuffle(sentences)
    return sentences


def generate_mixed_sequence_with_exhaustive(
        length: int, vocab: VocabConfig, rng: random.Random,
        p_relation: float, train_triples: Optional[List],
        p_reverse_short: float,
        contrastive_pool: List[List[int]], pool_idx: List[int]) -> List[int]:
    """Like generate_mixed_sequence but draws contrastive sentences from a pre-built pool.

    pool_idx is a single-element list holding the current index into contrastive_pool
    (mutable so it persists across calls).
    """
    all_entities = ([vocab.subject_id(i) for i in range(vocab.n_entities)] +
                    [vocab.object_id(j) for j in range(vocab.n_entities)])
    p_contrastive = len(contrastive_pool) / (len(contrastive_pool) + length / ATTR_SENTENCE_LEN * 100)
    # Aim for ~1 contrastive per sequence on average given pool size
    target_contrastive_frac = len(contrastive_pool) / (len(contrastive_pool) + (1 - p_relation - p_reverse_short) * 1000)
    tokens: List[int] = []
    while True:
        remaining = length - len(tokens)
        if remaining < ATTR_SENTENCE_LEN:
            break
        r = rng.random()
        if r < p_relation and remaining >= REL_SENTENCE_LEN:
            tokens.extend(generate_rel_sentence(vocab, rng, train_triples))
        elif r < p_relation + p_reverse_short and remaining >= REVERSE_SHORT_SENTENCE_LEN:
            tokens.extend(generate_reverse_short_sentence(vocab, rng))
        elif r < p_relation + p_reverse_short + 0.20 and remaining >= CONTRASTIVE_SENTENCE_LEN:
            idx = pool_idx[0] % len(contrastive_pool)
            tokens.extend(contrastive_pool[idx])
            pool_idx[0] += 1
        else:
            entity = rng.choice(all_entities)
            tokens.extend(generate_attr_sentence(entity, vocab, rng))
    while len(tokens) < length:
        tokens.append(vocab.sep_id)
    return tokens


class SequenceDataset(Dataset):
    def __init__(self, num_sequences: int, seq_length: int, vocab: VocabConfig,
                 seed: int = 0, p_relation: float = 0.0,
                 train_triples: Optional[List] = None,
                 p_reverse: float = 0.0,
                 p_reverse_short: float = 0.0,
                 p_contrastive: float = 0.0,
                 extra_sequences: Optional[List[List[int]]] = None,
                 contrastive_pool: Optional[List[List[int]]] = None):
        self.vocab = vocab
        rng = random.Random(seed)
        if contrastive_pool is not None:
            pool_idx = [0]
            rng_pool = random.Random(seed + 777)
            rng_pool.shuffle(contrastive_pool)

            def _next_contrastive():
                idx = pool_idx[0] % len(contrastive_pool)
                pool_idx[0] += 1
                return contrastive_pool[idx]

            self.sequences = [
                self._generate_with_pool(seq_length, vocab, rng, p_relation,
                                         train_triples, p_reverse_short,
                                         p_contrastive, _next_contrastive)
                for _ in range(num_sequences)
            ]
        elif p_relation > 0 or p_reverse > 0 or p_reverse_short > 0 or p_contrastive > 0:
            self.sequences = [
                generate_mixed_sequence(seq_length, vocab, rng, p_relation,
                                        train_triples, p_reverse, p_reverse_short,
                                        p_contrastive)
                for _ in range(num_sequences)
            ]
        else:
            self.sequences = [
                generate_sequence(seq_length, vocab, rng)
                for _ in range(num_sequences)
            ]
        if extra_sequences:
            self.sequences.extend(extra_sequences)

    @staticmethod
    def _generate_with_pool(length, vocab, rng, p_relation, train_triples,
                            p_reverse_short, p_contrastive, next_contrastive_fn):
        all_entities = ([vocab.subject_id(i) for i in range(vocab.n_entities)] +
                        [vocab.object_id(j) for j in range(vocab.n_entities)])
        tokens: List[int] = []
        while True:
            remaining = length - len(tokens)
            if remaining < ATTR_SENTENCE_LEN:
                break
            r = rng.random()
            if r < p_relation and remaining >= REL_SENTENCE_LEN:
                tokens.extend(generate_rel_sentence(vocab, rng, train_triples))
            elif r < p_relation + p_contrastive and remaining >= CONTRASTIVE_SENTENCE_LEN:
                tokens.extend(next_contrastive_fn())
            elif r < p_relation + p_contrastive + p_reverse_short and remaining >= REVERSE_SHORT_SENTENCE_LEN:
                tokens.extend(generate_reverse_short_sentence(vocab, rng))
            else:
                entity = rng.choice(all_entities)
                tokens.extend(generate_attr_sentence(entity, vocab, rng))
        while len(tokens) < length:
            tokens.append(vocab.sep_id)
        return tokens

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.sequences[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def make_dataloader(cfg: Config, seed_offset: int = 0, p_relation: float = 0.0,
                    train_triples: Optional[List] = None,
                    p_reverse: float = 0.0,
                    p_reverse_short: float = 0.0,
                    p_contrastive: float = 0.0,
                    extra_sequences: Optional[List[List[int]]] = None,
                    contrastive_pool: Optional[List[List[int]]] = None) -> DataLoader:
    ds = SequenceDataset(
        num_sequences=cfg.train.num_sequences,
        seq_length=cfg.train.seq_length,
        vocab=cfg.vocab,
        seed=cfg.seed + seed_offset,
        p_relation=p_relation,
        train_triples=train_triples,
        p_reverse=p_reverse,
        p_reverse_short=p_reverse_short,
        p_contrastive=p_contrastive,
        extra_sequences=extra_sequences,
        contrastive_pool=contrastive_pool,
    )
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)


def token_type_mask(tokens: torch.Tensor, vocab: VocabConfig):
    """Return boolean masks for each token type.

    Returns: (sep_mask, entity_mask, attr_mask, rel_mask)
    """
    sep_mask = tokens == vocab.sep_id
    entity_mask = (tokens >= vocab.subject_offset) & (tokens < vocab.f_attr_offset)
    attr_mask = (tokens >= vocab.f_attr_offset) & (tokens < vocab.relation_offset)
    rel_mask = (tokens >= vocab.relation_offset) & (tokens < vocab.vocab_size)
    return sep_mask, entity_mask, attr_mask, rel_mask
