"""
Standalone CLI demo for the AI Research KG RAG pipeline.

Runs a single question through baseline + RAG and prints the comparison.
For the interactive loop, use rag.py directly.

Usage:
  python src/rag/demo.py
  python src/rag/demo.py --question "What AI models are in the knowledge graph?"
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEMO_QUESTIONS = [
    "What AI models are in the knowledge graph?",
    "Which organizations are present in the knowledge graph?",
    "What is the source URL of BERT?",
]


def run_demo(question: str = None) -> None:
    from src.rag.rag import (
        check_ollama, load_graph, build_schema_summary,
        answer_baseline, answer_with_rag, pretty_print,
    )

    print("=" * 60)
    print("  AI Research KG — RAG Demo")
    print("=" * 60)

    model = check_ollama()
    g = load_graph()
    schema = build_schema_summary(g)
    print(f"\n[Ready] model={model}, triples={len(g)}, schema={len(schema)} chars\n")

    questions = [question] if question else DEMO_QUESTIONS

    for q in questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {q}")
        print("=" * 60)

        print("\n[Baseline — no KG]")
        print(answer_baseline(q, model=model))

        print("\n[RAG — NL -> SPARQL -> rdflib]")
        result = answer_with_rag(g, schema, q, model=model)
        pretty_print(result)

    print("\nDemo complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG demo for AI Research KG")
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Single question to answer (default: runs 3 sample questions)")
    args = parser.parse_args()
    run_demo(question=args.question)
