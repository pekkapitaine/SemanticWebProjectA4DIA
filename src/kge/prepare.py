"""
KGE Data Preparation.

Reads kg_artifacts/expanded.nt and produces clean train/valid/test splits
in tab-separated (head, relation, tail) format for PyKEEN.

Steps:
  1. Keep only URI-to-URI triples (drop literal objects)
  2. Filter predicates to top-N by frequency (keeps 50-200 relations)
  3. Ensure all entities in valid/test appear in train
  4. 80/10/10 split
  5. Save data/kge/train.txt, valid.txt, test.txt + stats

Output: data/kge/train.txt, valid.txt, test.txt
"""

import random
from collections import Counter
from pathlib import Path

from rdflib import Graph, URIRef

EXPANDED_PATH = Path(__file__).parent.parent.parent / "kg_artifacts" / "expanded.nt"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "kge"

# Keep only predicates with at least MIN_FREQ occurrences
MIN_PRED_FREQ = 50
# Max number of predicates to keep (target: 50-200)
MAX_PREDICATES = 150

SEED = 42


def _short(uri: str) -> str:
    """Return a short readable name for a URI."""
    u = uri.rstrip("/").rstrip("#")
    name = u.split("/")[-1].split("#")[-1]
    return name if name else uri


def prepare(
    expanded_path: Path = EXPANDED_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    random.seed(SEED)
    print(f"Loading {expanded_path} ...")
    g = Graph()
    g.parse(str(expanded_path), format="nt")

    # Step 1: keep URI-to-URI triples only
    triples = []
    pred_counter: Counter = Counter()
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef):
            ps = _short(str(p))
            triples.append((_short(str(s)), ps, _short(str(o))))
            pred_counter[ps] += 1

    print(f"  URI-URI triples: {len(triples):,}")

    # Step 2: filter predicates
    allowed_preds = {
        pred for pred, cnt in pred_counter.most_common(MAX_PREDICATES)
        if cnt >= MIN_PRED_FREQ
    }
    triples = [t for t in triples if t[1] in allowed_preds]
    print(f"  After predicate filter ({len(allowed_preds)} relations, min_freq={MIN_PRED_FREQ}): {len(triples):,} triples")

    # Step 3: deduplicate
    triples = list(set(triples))
    print(f"  After dedup: {len(triples):,} triples")

    # Step 4: ensure connectivity — remove entities appearing only once
    entity_count: Counter = Counter()
    for h, r, t in triples:
        entity_count[h] += 1
        entity_count[t] += 1
    triples = [t for t in triples if entity_count[t[0]] > 1 and entity_count[t[2]] > 1]
    print(f"  After connectivity filter: {len(triples):,} triples")

    # Step 5: split 80/10/10 with no entity leakage
    random.shuffle(triples)
    train_entities: set[str] = set()

    # First pass: put everything in train
    n = len(triples)
    cut1 = int(n * 0.80)
    cut2 = int(n * 0.90)

    train = triples[:cut1]
    val_raw = triples[cut1:cut2]
    test_raw = triples[cut2:]

    for h, r, t in train:
        train_entities.add(h)
        train_entities.add(t)

    # Move triples with unseen entities back to train
    valid, test = [], []
    for h, r, t in val_raw:
        if h in train_entities and t in train_entities:
            valid.append((h, r, t))
        else:
            train.append((h, r, t))
            train_entities.add(h)
            train_entities.add(t)

    for h, r, t in test_raw:
        if h in train_entities and t in train_entities:
            test.append((h, r, t))
        else:
            train.append((h, r, t))
            train_entities.add(h)
            train_entities.add(t)

    print(f"\n  Train : {len(train):,}")
    print(f"  Valid : {len(valid):,}")
    print(f"  Test  : {len(test):,}")
    print(f"  Entities in train: {len(train_entities):,}")
    print(f"  Relations: {len(allowed_preds)}")

    # Step 6: save
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in [("train", train), ("valid", valid), ("test", test)]:
        path = output_dir / f"{split_name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for h, r, t in split_data:
                f.write(f"{h}\t{r}\t{t}\n")
        print(f"  Saved {path}")

    # Save relation list for reference
    rel_path = output_dir / "relations.txt"
    with open(rel_path, "w", encoding="utf-8") as f:
        for pred, cnt in pred_counter.most_common(MAX_PREDICATES):
            if pred in allowed_preds:
                f.write(f"{pred}\t{cnt}\n")

    stats = {
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "entities": len(train_entities),
        "relations": len(allowed_preds),
    }
    return stats


if __name__ == "__main__":
    prepare()
