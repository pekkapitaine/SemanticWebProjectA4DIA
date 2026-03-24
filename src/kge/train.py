"""
KGE Training — TransE and DistMult via PyKEEN.

Trains two models on three KB sizes (20k / 50k / full) for size-sensitivity analysis.
Saves trained models + results to data/kge/results/.

Usage:
    python src/kge/train.py
    python src/kge/train.py --size 20k   # single size
    python src/kge/train.py --models transe  # single model
"""

import argparse
import json
import random
from pathlib import Path

import torch

KGE_DIR = Path(__file__).parent.parent.parent / "data" / "kge"
RESULTS_DIR = KGE_DIR / "results"

SEED = 42

# Training hyperparameters (consistent across models for fair comparison)
CONFIG = {
    "embedding_dim": 100,
    "num_epochs": 100,
    "batch_size": 512,
    "learning_rate": 0.01,
    "negative_sampler": "basic",
    "num_negs_per_pos": 64,
}

SIZES = {
    "20k":  20_000,
    "50k":  50_000,
    "full": None,   # use all triples
}


def _load_triples(path: Path) -> list[tuple[str, str, str]]:
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def _subsample(train: list, n: int | None, seed: int = SEED) -> list:
    if n is None or n >= len(train):
        return train
    rng = random.Random(seed)
    return rng.sample(train, n)


def train_model(model_name: str, size_label: str, size_limit: int | None) -> dict:
    """Train one model on one KB size. Returns metrics dict."""
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    print(f"\n{'='*55}")
    print(f"  Model: {model_name.upper()}  |  Size: {size_label}")
    print(f"{'='*55}")

    train_raw = _load_triples(KGE_DIR / "train.txt")
    valid_raw = _load_triples(KGE_DIR / "valid.txt")
    test_raw  = _load_triples(KGE_DIR / "test.txt")

    # Subsample train for size sensitivity
    train_sub = _subsample(train_raw, size_limit)

    # Build entity/relation vocabulary from subsampled train
    train_ents = {h for h, _, _ in train_sub} | {t for _, _, t in train_sub}
    train_rels = {r for _, r, _ in train_sub}

    # Filter valid/test to only include known entities
    valid_f = [(h, r, t) for h, r, t in valid_raw
               if h in train_ents and t in train_ents and r in train_rels]
    test_f  = [(h, r, t) for h, r, t in test_raw
               if h in train_ents and t in train_ents and r in train_rels]

    print(f"  Train triples : {len(train_sub):,}")
    print(f"  Valid triples : {len(valid_f):,}")
    print(f"  Test triples  : {len(test_f):,}")

    if len(train_sub) < 500 or len(valid_f) < 10 or len(test_f) < 10:
        print("  [SKIP] Not enough triples for this configuration.")
        return {}

    # Build TriplesFactory
    import numpy as np
    tf_train = TriplesFactory.from_labeled_triples(
        triples=np.array([[h, r, t] for h, r, t in train_sub]),
        create_inverse_triples=False,
    )
    tf_valid = TriplesFactory.from_labeled_triples(
        triples=np.array([[h, r, t] for h, r, t in valid_f]),
        entity_to_id=tf_train.entity_to_id,
        relation_to_id=tf_train.relation_to_id,
        create_inverse_triples=False,
    )
    tf_test = TriplesFactory.from_labeled_triples(
        triples=np.array([[h, r, t] for h, r, t in test_f]),
        entity_to_id=tf_train.entity_to_id,
        relation_to_id=tf_train.relation_to_id,
        create_inverse_triples=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    result = pipeline(
        training=tf_train,
        validation=tf_valid,
        testing=tf_test,
        model=model_name,
        model_kwargs={"embedding_dim": CONFIG["embedding_dim"]},
        optimizer="Adam",
        optimizer_kwargs={"lr": CONFIG["learning_rate"]},
        training_kwargs={
            "num_epochs": CONFIG["num_epochs"],
            "batch_size": CONFIG["batch_size"],
        },
        negative_sampler=CONFIG["negative_sampler"],
        negative_sampler_kwargs={"num_negs_per_pos": CONFIG["num_negs_per_pos"]},
        evaluator_kwargs={"filtered": True},
        random_seed=SEED,
        device=device,
    )

    metrics = result.metric_results.to_dict()
    both = metrics.get("both", {}).get("realistic", {})

    summary = {
        "model": model_name,
        "size": size_label,
        "train_triples": len(train_sub),
        "mrr":    round(both.get("inverse_harmonic_mean_rank", 0), 4),
        "hits@1": round(both.get("hits_at_1", 0), 4),
        "hits@3": round(both.get("hits_at_3", 0), 4),
        "hits@10":round(both.get("hits_at_10", 0), 4),
    }
    print(f"\n  Results: MRR={summary['mrr']:.4f} | H@1={summary['hits@1']:.4f} "
          f"| H@3={summary['hits@3']:.4f} | H@10={summary['hits@10']:.4f}")

    # Save model
    out_dir = RESULTS_DIR / f"{model_name}_{size_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.save_to_directory(str(out_dir))

    return summary


def run_all(models: list[str], sizes: list[str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for size_label in sizes:
        size_limit = SIZES[size_label]
        for model_name in models:
            summary = train_model(model_name, size_label, size_limit)
            if summary:
                all_results.append(summary)

    # Print summary table
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<12} {'Size':<8} {'Train':>8} {'MRR':>8} {'H@1':>8} {'H@3':>8} {'H@10':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"  {r['model']:<12} {r['size']:<8} {r['train_triples']:>8,} "
              f"{r['mrr']:>8.4f} {r['hits@1']:>8.4f} {r['hits@3']:>8.4f} {r['hits@10']:>8.4f}")

    # Save results JSON
    results_path = RESULTS_DIR / "results_summary.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"config": CONFIG, "results": all_results}, f, indent=2)
    print(f"\n  Saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["transe", "distmult"],
                        choices=["transe", "distmult", "complex", "rotate"])
    parser.add_argument("--sizes", nargs="+", default=["20k", "50k", "full"],
                        choices=list(SIZES.keys()))
    args = parser.parse_args()
    run_all(args.models, args.sizes)
