"""
Gradio UI for the AI Research KG RAG pipeline.

Provides a web interface with two tabs:
  1. RAG Query — NL question -> SPARQL -> results
  2. Baseline — direct LLM answer without KG

Requires: Ollama running locally (ollama serve)
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import gradio as gr

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Lazy-loaded globals (loaded once on first query)
# ---------------------------------------------------------------------------

_graph = None
_schema = None
_model = None
_load_error = None


def _ensure_loaded():
    global _graph, _schema, _model, _load_error
    if _graph is not None:
        return True
    try:
        from src.rag.rag import check_ollama, load_graph, build_schema_summary
        _model = check_ollama()
        _graph = load_graph()
        _schema = build_schema_summary(_graph)
        _load_error = None
        return True
    except Exception as e:
        _load_error = str(e)
        return False


# ---------------------------------------------------------------------------
# Query handlers
# ---------------------------------------------------------------------------

def query_rag(question: str):
    """Run the full RAG pipeline and return formatted results."""
    if not question.strip():
        return "Please enter a question.", "", ""

    if not _ensure_loaded():
        return f"Error loading KG/Ollama: {_load_error}", "", ""

    from src.rag.rag import answer_with_rag
    result = answer_with_rag(_graph, _schema, question, model=_model)

    sparql_query = result.get("query", "N/A")
    rows = result.get("rows", [])
    vars_ = result.get("vars", [])
    repaired = result.get("repaired", False)
    error = result.get("error")

    # Build results text
    if error and not rows:
        results_text = f"Error executing query:\n{error}"
    elif not rows:
        results_text = "No results returned."
    else:
        header = " | ".join(vars_)
        sep = "-" * max(len(header), 40)
        lines = [header, sep]
        for row in rows[:20]:
            lines.append(" | ".join(row))
        if len(rows) > 20:
            lines.append(f"... ({len(rows)} total rows)")
        results_text = "\n".join(lines)

    repair_note = " (query was auto-repaired)" if repaired else ""
    status = f"Model: {_model} | Graph: {len(_graph)} triples{repair_note}"

    return sparql_query, results_text, status


def query_baseline(question: str):
    """Run baseline LLM without KG."""
    if not question.strip():
        return "Please enter a question."

    if not _ensure_loaded():
        return f"Error loading Ollama: {_load_error}"

    from src.rag.rag import answer_baseline
    return answer_baseline(question, model=_model)


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="AI Research KG — RAG Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# AI Research Knowledge Graph — RAG Demo
Query the AI Research Knowledge Graph using natural language.
The RAG pipeline converts your question to SPARQL and retrieves grounded answers.
        """)

        with gr.Tabs():
            # ---- Tab 1: RAG ----
            with gr.Tab("RAG (NL -> SPARQL)"):
                gr.Markdown("Ask a question — the system will generate and execute a SPARQL query against the KG.")

                rag_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. What AI models are in the knowledge graph?",
                    lines=2,
                )
                rag_btn = gr.Button("Ask (RAG)", variant="primary")

                with gr.Row():
                    sparql_out = gr.Code(
                        label="Generated SPARQL",
                        language="sql",
                        lines=8,
                    )
                    results_out = gr.Textbox(
                        label="Query Results",
                        lines=10,
                    )

                status_out = gr.Textbox(label="Status", interactive=False, lines=1)

                rag_btn.click(
                    fn=query_rag,
                    inputs=rag_input,
                    outputs=[sparql_out, results_out, status_out],
                )

                gr.Examples(
                    examples=[
                        ["What AI models are in the knowledge graph?"],
                        ["Which organizations are present in the knowledge graph?"],
                        ["Who are the researchers mentioned in the knowledge graph?"],
                        ["What is the source URL of BERT?"],
                        ["Which entities are of type AIModel?"],
                    ],
                    inputs=rag_input,
                )

            # ---- Tab 2: Baseline ----
            with gr.Tab("Baseline (LLM only)"):
                gr.Markdown("Ask the same question without knowledge graph grounding — for comparison.")

                baseline_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. What AI models exist?",
                    lines=2,
                )
                baseline_btn = gr.Button("Ask (Baseline)", variant="secondary")
                baseline_out = gr.Textbox(label="LLM Answer", lines=10)

                baseline_btn.click(
                    fn=query_baseline,
                    inputs=baseline_input,
                    outputs=baseline_out,
                )

            # ---- Tab 3: About ----
            with gr.Tab("About"):
                gr.Markdown("""
## About this Demo

**Knowledge Graph**: AI Research KB built from HuggingFace blog crawling + NER extraction.
- ~1,369 triples (initial_kg.ttl) + alignment triples (alignment.ttl)
- Classes: AIModel, Organization, Researcher, Dataset, Benchmark, Technique
- Expanded KG: ~85k triples via Wikidata SPARQL

**RAG Pipeline**:
1. User question -> Ollama LLM generates SPARQL query
2. Query executed against rdflib graph
3. If query fails: self-repair loop (up to 2 attempts)
4. Results returned as structured data

**Models supported** (auto-detected): gemma:2b, gemma2:2b, mistral, llama3.2:1b

**Baseline**: Direct LLM answer without KG access — shows hallucination risk.
                """)

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(share=args.share, server_port=args.port)
