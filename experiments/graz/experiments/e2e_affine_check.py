"""
End-to-end affine R² check: subject embedding → final hidden state.

For each relation, fit an affine map from subject token embeddings
to the model's final hidden state at the prediction position.
This tells us whether the OVERALL subject→object transformation
is affine, independent of how it's internally implemented.

Also checks: subject embedding → object embedding (pure embedding space).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from src.config import ModelConfig
from src.model import build_model
from experiments.ravfogel_1layer import RavfogelVocab

ROOT = Path(__file__).parent.parent

ACT_FNS = {"gelu": F.gelu, "relu": F.relu}

def patch_activation(model, act_name):
    import types
    act_fn = ACT_FNS[act_name]
    for block in model.blocks:
        if hasattr(block, 'mlp'):
            fc1 = block.mlp.fc1
            fc2 = block.mlp.fc2
            def make_forward(fc1, fc2, fn):
                def forward(self, x):
                    return fc2(fn(fc1(x)))
                return forward
            block.mlp.forward = types.MethodType(make_forward(fc1, fc2, act_fn), block.mlp)

def load_model(activation, dim=64, seed=42):
    act_suffix = f"_{activation}" if activation != "gelu" else ""
    ckpt_path = ROOT / f"checkpoints_ravfogel_cluster_dim{dim}_seed{seed}{act_suffix}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    vocab = RavfogelVocab(n_entities=ckpt["n_entities"], num_relations=2)
    for k, p in enumerate(ckpt["permutations"]):
        vocab.set_permutation(k, p)

    cfg = ckpt["model_cfg"]
    model_cfg = ModelConfig(n_layers=cfg["n_layers"], hidden_dim=cfg["hidden_dim"],
                            n_heads=cfg["n_heads"])
    model = build_model(model_cfg, vocab)
    if activation != "gelu":
        patch_activation(model, activation)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab

def fit_affine_r2(X, Y):
    n = X.shape[0]
    X_aug = torch.cat([X, torch.ones(n, 1)], dim=1)
    sol, _, _, _ = torch.linalg.lstsq(X_aug, Y)
    Y_pred = X_aug @ sol
    ss_res = ((Y - Y_pred) ** 2).sum().item()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum().item()
    return 1 - ss_res / max(ss_tot, 1e-10)

@torch.no_grad()
def extract_representations(model, vocab, batch_size=512):
    """Extract subject embeddings, object embeddings, and final hidden states."""
    n_ent = vocab.n_entities
    n_rel = vocab.num_relations

    results = {}
    for k in range(n_rel):
        subj_embeds = []
        obj_embeds = []
        final_hidden = []

        all_entities = list(range(n_ent))
        for start in range(0, n_ent, batch_size):
            batch_ents = all_entities[start:start+batch_size]
            # Input: [sep, relation, subject] — predict object at position 2
            contexts = torch.tensor(
                [[vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)]
                 for i in batch_ents], dtype=torch.long)

            # Get subject token embeddings (no positional encoding)
            s_emb = model.token_emb(contexts[:, 2])  # (B, dim)
            subj_embeds.append(s_emb)

            # Get object token embeddings
            obj_ids = torch.tensor([vocab.object_id(vocab.relation_object(i, k))
                                    for i in batch_ents], dtype=torch.long)
            o_emb = model.token_emb(obj_ids)  # (B, dim)
            obj_embeds.append(o_emb)

            # Run full forward pass, capture residual stream
            with model.capture_residuals() as residuals:
                logits = model(contexts)

            # Final hidden state at position 2 (after ln_f, before lm_head)
            # residuals[-1] is the output of the last block (before ln_f)
            final_h = model.ln_f(residuals[-1][:, 2, :])  # (B, dim)
            final_hidden.append(final_h)

        results[k] = {
            "subj_emb": torch.cat(subj_embeds, dim=0),
            "obj_emb": torch.cat(obj_embeds, dim=0),
            "final_hidden": torch.cat(final_hidden, dim=0),
        }
    return results

def main():
    print("=" * 70)
    print("END-TO-END AFFINE R² CHECK")
    print("=" * 70)

    for act_name in ["gelu", "relu"]:
        print(f"\n{'='*70}")
        print(f"Activation: {act_name.upper()}")
        print(f"{'='*70}")

        model, vocab = load_model(act_name)
        reps = extract_representations(model, vocab)

        for k in range(vocab.num_relations):
            S = reps[k]["subj_emb"]
            O = reps[k]["obj_emb"]
            H = reps[k]["final_hidden"]

            # 1. Subject embedding → final hidden state
            r2_s2h = fit_affine_r2(S, H)

            # 2. Subject embedding → object embedding
            r2_s2o = fit_affine_r2(S, O)

            # 3. Final hidden state → object embedding
            r2_h2o = fit_affine_r2(H, O)

            # 4. Check prediction accuracy via logits
            with torch.no_grad():
                logits = model.lm_head(H)  # (n_ent, vocab_size)
                preds = logits.argmax(dim=-1)
                obj_ids = torch.tensor([vocab.object_id(vocab.relation_object(i, k))
                                        for i in range(vocab.n_entities)])
                acc = (preds == obj_ids).float().mean().item()

            print(f"\n  Relation {k} (acc={acc:.4f}):")
            print(f"    subject_emb → final_hidden   R² = {r2_s2h:.4f}")
            print(f"    subject_emb → object_emb     R² = {r2_s2o:.4f}")
            print(f"    final_hidden → object_emb    R² = {r2_h2o:.4f}")

        # Also: combined across relations
        S_all = torch.cat([reps[k]["subj_emb"] for k in range(vocab.num_relations)])
        O_all = torch.cat([reps[k]["obj_emb"] for k in range(vocab.num_relations)])
        H_all = torch.cat([reps[k]["final_hidden"] for k in range(vocab.num_relations)])

        r2_combined_s2h = fit_affine_r2(S_all, H_all)
        r2_combined_s2o = fit_affine_r2(S_all, O_all)
        r2_combined_h2o = fit_affine_r2(H_all, O_all)

        print(f"\n  Combined (both relations):")
        print(f"    subject_emb → final_hidden   R² = {r2_combined_s2h:.4f}")
        print(f"    subject_emb → object_emb     R² = {r2_combined_s2o:.4f}")
        print(f"    final_hidden → object_emb    R² = {r2_combined_h2o:.4f}")

if __name__ == "__main__":
    main()

def main_v2():
    """
    Better framing: the model's job is subject_emb → W_u^T object_emb,
    where W_u is the unembedding. With tied weights, W_u = W_e, so the
    model needs final_hidden such that final_hidden @ W_e^T has max at
    the object token.

    But the Ravfogel question is really: is there an affine map A such that
    A @ subject_representation ≈ object_representation?

    The right "representations" are the internal ones (residual stream),
    not the raw token embeddings. Let's check:
    - subject representation = residual stream at subject position after layer 0
    - object representation = the "ideal" vector that the unembedding would map to
      the object token = the object's row in the unembedding matrix = object embedding

    Actually, the cleanest test: does final_hidden ≈ A @ mlp_input?
    That's what the cluster analysis already measures.

    NEW check: does the lm_head(final_hidden) logit vector relate affinely
    to lm_head(subject_emb)? i.e., is the logit-space mapping affine?
    """
    print("\n" + "=" * 70)
    print("LOGIT-SPACE AFFINE CHECK")
    print("=" * 70)

    for act_name in ["gelu", "relu"]:
        print(f"\n--- {act_name.upper()} ---")
        model, vocab = load_model(act_name)
        reps = extract_representations(model, vocab)

        for k in range(vocab.num_relations):
            H = reps[k]["final_hidden"]  # (n_ent, dim)
            S = reps[k]["subj_emb"]

            # Logits from final hidden
            logits_h = model.lm_head(H)  # (n_ent, vocab_size)
            # Logits from subject embedding (what would lm_head give for raw subject?)
            logits_s = model.lm_head(S)  # (n_ent, vocab_size)

            # Check: is logits_h ≈ A @ logits_s + b? (logit-space affine)
            r2_logit = fit_affine_r2(logits_s, logits_h)
            print(f"  R{k}: logits(subject_emb) → logits(final_hidden)  R² = {r2_logit:.4f}")

            # The unembedding is just the embedding matrix transposed.
            # final_hidden @ E^T = logits. So if final_hidden = A @ subj_emb + b,
            # then logits = (A @ subj_emb + b) @ E^T = subj_emb @ A^T @ E^T + b @ E^T
            # = (A^T @ E^T)^T @ subj_emb^T + const... it's still affine in logit space.
            # So logit-space R² should equal hidden-space R². Let's verify.

    print("\n" + "=" * 70)
    print("RESIDUAL-STREAM AFFINE CHECK (input → output of full block)")
    print("=" * 70)
    print("(This is what the MLP cluster analysis measures, for reference)")

    for act_name in ["gelu", "relu"]:
        print(f"\n--- {act_name.upper()} ---")
        model, vocab = load_model(act_name)

        n_ent = vocab.n_entities
        for k in range(vocab.num_relations):
            # Get residual stream before and after the transformer block
            pre_block = []
            post_block = []

            for start in range(0, n_ent, 512):
                batch_ents = list(range(start, min(start+512, n_ent)))
                contexts = torch.tensor(
                    [[vocab.sep_id, vocab.relation_id(k), vocab.subject_id(i)]
                     for i in batch_ents], dtype=torch.long)

                with model.capture_residuals() as residuals:
                    _ = model(contexts)

                # residuals[0] = embedding output (before block)
                # residuals[1] = after block 0
                pre_block.append(residuals[0][:, 2, :])
                post_block.append(residuals[1][:, 2, :])

            pre = torch.cat(pre_block, dim=0)
            post = torch.cat(post_block, dim=0)

            r2 = fit_affine_r2(pre, post)
            print(f"  R{k}: pre-block residual → post-block residual  R² = {r2:.4f}")

main_v2()
