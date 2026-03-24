"""
KGE Evaluation — Nearest Neighbors, t-SNE, Relation Analysis.

Loads the best trained model (DistMult full) and produces:
  - Nearest neighbor analysis for selected AI entities
  - t-SNE 2D scatter plot colored by entity type
  - Relation behavior analysis

Output: data/kge/results/tsne_plot.png, nearest_neighbors.txt
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "kge" / "results"
KGE_DIR = Path(__file__).parent.parent.parent / "data" / "kge"
OUTPUT_DIR = RESULTS_DIR


def _load_best_model():
    """Load the DistMult full model (best performer) by re-running the pipeline."""
    import torch
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train import _load_triples, CONFIG, SEED

    train_raw = _load_triples(KGE_DIR / "train.txt")
    valid_raw = _load_triples(KGE_DIR / "valid.txt")
    test_raw  = _load_triples(KGE_DIR / "test.txt")

    train_ents = {h for h, _, _ in train_raw} | {t for _, _, t in train_raw}
    train_rels = {r for _, r, _ in train_raw}
    valid_f = [(h, r, t) for h, r, t in valid_raw if h in train_ents and t in train_ents and r in train_rels]
    test_f  = [(h, r, t) for h, r, t in test_raw  if h in train_ents and t in train_ents and r in train_rels]

    tf_train = TriplesFactory.from_labeled_triples(
        triples=np.array([[h, r, t] for h, r, t in train_raw]), create_inverse_triples=False)
    tf_valid = TriplesFactory.from_labeled_triples(
        triples=np.array([[h, r, t] for h, r, t in valid_f]),
        entity_to_id=tf_train.entity_to_id, relation_to_id=tf_train.relation_to_id, create_inverse_triples=False)
    tf_test = TriplesFactory.from_labeled_triples(
        triples=np.array([[h, r, t] for h, r, t in test_f]),
        entity_to_id=tf_train.entity_to_id, relation_to_id=tf_train.relation_to_id, create_inverse_triples=False)

    result = pipeline(
        training=tf_train, validation=tf_valid, testing=tf_test,
        model="distmult",
        model_kwargs={"embedding_dim": CONFIG["embedding_dim"]},
        optimizer="Adam", optimizer_kwargs={"lr": CONFIG["learning_rate"]},
        training_kwargs={"num_epochs": CONFIG["num_epochs"], "batch_size": CONFIG["batch_size"]},
        negative_sampler=CONFIG["negative_sampler"],
        negative_sampler_kwargs={"num_negs_per_pos": CONFIG["num_negs_per_pos"]},
        evaluator_kwargs={"filtered": True},
        random_seed=SEED, device="cpu",
    )
    return result


def nearest_neighbors(result, entities: list[str], k: int = 5) -> dict:
    """Find k nearest neighbors for each entity in embedding space."""
    import torch

    model = result.model
    entity_to_id = result.training.entity_to_id

    # Get embedding matrix
    embeddings = model.entity_representations[0](
        indices=None
    ).detach().cpu().numpy()  # (num_entities, dim)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    emb_norm = embeddings / norms

    results = {}
    for ent in entities:
        if ent not in entity_to_id:
            # Try partial match
            matches = [k for k in entity_to_id if ent.lower() in k.lower()]
            if matches:
                ent_key = matches[0]
            else:
                results[ent] = f"Entity '{ent}' not found in vocabulary"
                continue
        else:
            ent_key = ent

        eid = entity_to_id[ent_key]
        q = emb_norm[eid]
        sims = emb_norm @ q
        sims[eid] = -1  # exclude self
        top_k = np.argsort(sims)[::-1][:k]

        id_to_entity = {v: k for k, v in entity_to_id.items()}
        neighbors = [(id_to_entity[idx], float(sims[idx])) for idx in top_k]
        results[ent_key] = neighbors

    return results


def run_tsne(result, n_entities: int = 2000) -> str:
    """Generate t-SNE plot of entity embeddings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model = result.model
    entity_to_id = result.training.entity_to_id

    embeddings = model.entity_representations[0](
        indices=None
    ).detach().cpu().numpy()

    # Sample entities
    n = min(n_entities, len(embeddings))
    idx = np.random.RandomState(42).choice(len(embeddings), n, replace=False)
    emb_sample = embeddings[idx]

    id_to_entity = {v: k for k, v in entity_to_id.items()}

    # Color by entity type (heuristic based on short name)
    def _color(name: str) -> str:
        n = name.lower()
        if any(k in n for k in ["Q", "P"]) and len(name) < 8:
            return "gray"
        if any(k in n for k in ["bert", "gpt", "llama", "falcon", "gemma",
                                  "mistral", "distil", "roberta", "t5", "starcoder"]):
            return "red"
        if any(k in n for k in ["google", "microsoft", "meta", "openai", "nvidia",
                                  "amazon", "anthropic", "deepmind", "hugging"]):
            return "blue"
        if any(k in n for k in ["P178", "P31", "P277", "P361", "P108"]):
            return "green"
        return "gray"

    labels = [id_to_entity.get(i, "?") for i in idx]
    colors = [_color(l) for l in labels]

    print("Running t-SNE (may take ~1 min)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    emb_2d = tsne.fit_transform(emb_sample)

    fig, ax = plt.subplots(figsize=(12, 10))
    unique_colors = list(set(colors))
    color_labels = {"red": "AI Models", "blue": "Organizations", "green": "Properties", "gray": "Other"}

    for c in unique_colors:
        mask = [i for i, col in enumerate(colors) if col == c]
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=c, s=8, alpha=0.6, label=color_labels.get(c, c)
        )

    ax.set_title("t-SNE of Entity Embeddings (DistMult, full KB)", fontsize=14)
    ax.legend(markerscale=3)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "tsne_plot.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"t-SNE plot saved to {out_path}")
    return str(out_path)


def analyze_relations(result) -> None:
    """Analyze relation embedding norms and symmetry."""
    model = result.model
    relation_to_id = result.training.relation_to_id

    rel_embs = model.relation_representations[0](
        indices=None
    ).detach().cpu().numpy()

    norms = np.linalg.norm(rel_embs, axis=1)
    id_to_rel = {v: k for k, v in relation_to_id.items()}

    print("\nRelation norms (top 10 by magnitude):")
    top_rels = np.argsort(norms)[::-1][:10]
    for idx in top_rels:
        print(f"  {id_to_rel[idx]:<25} norm={norms[idx]:.4f}")


def run_all():
    print("Loading best model (DistMult full)...")
    result = _load_best_model()

    # ---- Nearest neighbors ----
    print("\n=== Nearest Neighbor Analysis ===")
    target_entities = ["BERT", "Google", "Microsoft", "OpenAI", "Transformer", "PyTorch"]
    nn_results = nearest_neighbors(result, target_entities, k=5)

    nn_lines = ["Nearest Neighbor Analysis (DistMult, full KB)", "=" * 50, ""]
    for ent, neighbors in nn_results.items():
        nn_lines.append(f"Entity: {ent}")
        if isinstance(neighbors, str):
            nn_lines.append(f"  {neighbors}")
        else:
            for nb, sim in neighbors:
                nn_lines.append(f"  {nb:<35} cos_sim={sim:.4f}")
        nn_lines.append("")

    nn_text = "\n".join(nn_lines)
    print(nn_text)
    nn_path = OUTPUT_DIR / "nearest_neighbors.txt"
    nn_path.write_text(nn_text, encoding="utf-8")

    # ---- Relation analysis ----
    print("\n=== Relation Analysis ===")
    analyze_relations(result)

    # ---- t-SNE ----
    print("\n=== t-SNE Visualization ===")
    run_tsne(result, n_entities=1500)

    print("\nAll analysis complete.")


if __name__ == "__main__":
    run_all()
