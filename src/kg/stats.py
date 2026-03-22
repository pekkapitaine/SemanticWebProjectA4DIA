"""
KB Statistics — counts triples, entities, relations and prints a summary.

Reads kg_artifacts/expanded.nt and produces kg_artifacts/kb_stats.txt.
"""

from collections import Counter
from pathlib import Path

from rdflib import Graph, URIRef

EXPANDED_PATH = Path(__file__).parent.parent.parent / "kg_artifacts" / "expanded.nt"
STATS_PATH = Path(__file__).parent.parent.parent / "kg_artifacts" / "kb_stats.txt"


def compute_stats(expanded_path: Path = EXPANDED_PATH, stats_path: Path = STATS_PATH) -> dict:
    print(f"Loading {expanded_path} ...")
    g = Graph()
    g.parse(str(expanded_path), format="nt")

    subjects: set[str] = set()
    objects_uri: set[str] = set()
    predicates: Counter = Counter()

    for s, p, o in g:
        subjects.add(str(s))
        predicates[str(p)] += 1
        if isinstance(o, URIRef):
            objects_uri.add(str(o))

    entities = subjects | objects_uri
    n_triples = len(g)
    n_entities = len(entities)
    n_relations = len(predicates)

    top_predicates = predicates.most_common(20)

    lines = [
        "=" * 50,
        "  AI Research Knowledge Graph — KB Statistics",
        "=" * 50,
        f"  Triples   : {n_triples:,}",
        f"  Entities  : {n_entities:,}",
        f"  Relations : {n_relations:,}",
        "",
        "  Top 20 predicates:",
    ]
    for pred, count in top_predicates:
        short = pred.split("/")[-1]
        lines.append(f"    {short:<30} {count:>8,}")
    lines.append("=" * 50)

    report = "\n".join(lines)
    print(report)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(report, encoding="utf-8")
    print(f"\nStats saved to {stats_path}")

    return {
        "triples": n_triples,
        "entities": n_entities,
        "relations": n_relations,
    }


if __name__ == "__main__":
    compute_stats()
