"""
RAG pipeline: Natural Language -> SPARQL -> RDF results.

Adapted from the Lab 4 provided example, using our AI Research KB.

Features:
  - Load RDF graph (initial_kg.ttl + alignment.ttl merged)
  - Build schema summary (prefixes, predicates, classes, sample triples)
  - Baseline: direct LLM answer without KG
  - RAG: NL -> SPARQL (via Ollama) -> execute on rdflib graph
  - Self-repair: if SPARQL fails, ask LLM to fix it (up to MAX_REPAIRS attempts)
  - CLI demo loop
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

import requests
from rdflib import Graph

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KG_DIR = Path(__file__).parent.parent.parent / "kg_artifacts"
TTL_FILE   = KG_DIR / "initial_kg.ttl"
ALIGN_FILE = KG_DIR / "alignment.ttl"

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"   # fallback tried: gemma2:2b, mistral, llama3.2:1b

MAX_PREDICATES = 80
MAX_CLASSES    = 40
SAMPLE_TRIPLES = 20
MAX_REPAIRS    = 2

# ---------------------------------------------------------------------------
# 0) Utility: call local LLM via Ollama REST API
# ---------------------------------------------------------------------------

def ask_local_llm(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Send a prompt to Ollama and return the response string."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure it is running:\n"
            "  ollama serve\n"
            "  ollama run gemma:2b"
        )


def check_ollama(model: str = OLLAMA_MODEL) -> str:
    """Try the configured model; fall back to other small models if not found."""
    candidates = [model, "gemma2:2b", "mistral", "llama3.2:1b", "qwen:0.5b"]
    for m in candidates:
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": m, "prompt": "hi", "stream": False},
                timeout=30,
            )
            if resp.status_code == 200:
                print(f"[Ollama] Using model: {m}")
                return m
        except Exception:
            continue
    raise RuntimeError("No Ollama model available. Run: ollama pull gemma:2b")


# ---------------------------------------------------------------------------
# 1) Load RDF graph
# ---------------------------------------------------------------------------

def load_graph(ttl_path: Path = TTL_FILE, align_path: Path = ALIGN_FILE) -> Graph:
    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    if align_path.exists():
        g.parse(str(align_path), format="turtle")
    print(f"[Graph] Loaded {len(g)} triples from {ttl_path.name} + {align_path.name}")
    return g


# ---------------------------------------------------------------------------
# 2) Build schema summary
# ---------------------------------------------------------------------------

def get_prefix_block(g: Graph) -> str:
    defaults = {
        "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd":  "http://www.w3.org/2001/XMLSchema#",
        "owl":  "http://www.w3.org/2002/07/owl#",
        "kg":   "http://semanticweb.esilv.fr/aikg/",
    }
    ns_map = {p: str(ns) for p, ns in g.namespace_manager.namespaces()}
    for k, v in defaults.items():
        ns_map.setdefault(k, v)
    lines = [f"PREFIX {p}: <{ns}>" for p, ns in sorted(ns_map.items()) if p]
    return "\n".join(lines)


def list_distinct_predicates(g: Graph, limit: int = MAX_PREDICATES) -> List[str]:
    q = f"SELECT DISTINCT ?p WHERE {{ ?s ?p ?o . }} LIMIT {limit}"
    return [str(row.p) for row in g.query(q)]


def list_distinct_classes(g: Graph, limit: int = MAX_CLASSES) -> List[str]:
    q = f"SELECT DISTINCT ?cls WHERE {{ ?s a ?cls . }} LIMIT {limit}"
    return [str(row.cls) for row in g.query(q)]


def sample_triples(g: Graph, limit: int = SAMPLE_TRIPLES) -> List[Tuple[str, str, str]]:
    q = f"""
    SELECT ?s ?p ?o WHERE {{
      ?s ?p ?o .
      FILTER(isLiteral(?o))
    }} LIMIT {limit}
    """
    return [(str(r.s), str(r.p), str(r.o)) for r in g.query(q)]


def build_schema_summary(g: Graph) -> str:
    prefixes  = get_prefix_block(g)
    preds     = list_distinct_predicates(g)
    classes   = list_distinct_classes(g)
    samples   = sample_triples(g)

    pred_lines   = "\n".join(f"  - {p}" for p in preds)
    class_lines  = "\n".join(f"  - {c}" for c in classes)
    sample_lines = "\n".join(f"  - <{s}> <{p}> {repr(o)}" for s, p, o in samples)

    return f"""
{prefixes}

# Predicates (up to {MAX_PREDICATES}):
{pred_lines}

# Classes (up to {MAX_CLASSES}):
{class_lines}

# Sample triples (literals):
{sample_lines}
""".strip()


# ---------------------------------------------------------------------------
# 3) NL -> SPARQL prompting
# ---------------------------------------------------------------------------

SPARQL_INSTRUCTIONS = """\
You are a SPARQL generator for an AI Research Knowledge Graph.
Convert the QUESTION into a valid SPARQL 1.1 SELECT query.

Rules:
- Use ONLY the IRIs/prefixes visible in the SCHEMA SUMMARY.
- The main namespace is PREFIX kg: <http://semanticweb.esilv.fr/aikg/>
- Entities have rdfs:label literals (use FILTER + CONTAINS or direct URI).
- Prefer simple queries: SELECT ?s ?label WHERE { ?s a kg:AIModel ; rdfs:label ?label }
- Return ONLY the SPARQL query inside a single ```sparql ... ``` code block.
- No explanations outside the code block.
"""

CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sparql(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def make_sparql_prompt(schema_summary: str, question: str) -> str:
    return f"""{SPARQL_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema_summary}

QUESTION: {question}

Return only the SPARQL query in a ```sparql``` code block."""


def generate_sparql(question: str, schema_summary: str, model: str = OLLAMA_MODEL) -> str:
    raw = ask_local_llm(make_sparql_prompt(schema_summary, question), model=model)
    return extract_sparql(raw)


# ---------------------------------------------------------------------------
# 4) Execute SPARQL + self-repair
# ---------------------------------------------------------------------------

def run_sparql(g: Graph, query: str) -> Tuple[List[str], List[Tuple]]:
    res = g.query(query)
    vars_ = [str(v) for v in res.vars]
    rows  = [tuple(str(cell) for cell in r) for r in res]
    return vars_, rows


REPAIR_INSTRUCTIONS = """\
The previous SPARQL query failed. Using the SCHEMA SUMMARY and the ERROR MESSAGE,
return a corrected SPARQL 1.1 SELECT query.

Rules:
- Use only known prefixes/IRIs from the schema.
- Keep the query as simple as possible.
- Return ONLY the corrected SPARQL in a single ```sparql ... ``` code block.
"""


def repair_sparql(
    schema_summary: str, question: str, bad_query: str, error_msg: str,
    model: str = OLLAMA_MODEL
) -> str:
    prompt = f"""{REPAIR_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema_summary}

ORIGINAL QUESTION: {question}

BAD SPARQL:
{bad_query}

ERROR: {error_msg}

Return only the corrected SPARQL in a ```sparql``` code block."""
    raw = ask_local_llm(prompt, model=model)
    return extract_sparql(raw)


def answer_with_rag(
    g: Graph, schema_summary: str, question: str,
    model: str = OLLAMA_MODEL, try_repair: bool = True
) -> dict:
    """Full RAG pipeline: NL -> SPARQL -> execute -> optional repair."""
    sparql = generate_sparql(question, schema_summary, model=model)

    for attempt in range(MAX_REPAIRS + 1):
        try:
            vars_, rows = run_sparql(g, sparql)
            return {
                "query": sparql, "vars": vars_, "rows": rows,
                "repaired": attempt > 0, "repairs": attempt, "error": None,
            }
        except Exception as e:
            err = str(e)
            if not try_repair or attempt >= MAX_REPAIRS:
                return {
                    "query": sparql, "vars": [], "rows": [],
                    "repaired": attempt > 0, "repairs": attempt, "error": err,
                }
            sparql = repair_sparql(schema_summary, question, sparql, err, model=model)

    return {"query": sparql, "vars": [], "rows": [], "repaired": True, "repairs": MAX_REPAIRS, "error": "Max repairs reached"}


# ---------------------------------------------------------------------------
# 5) Baseline: direct LLM answer without KG
# ---------------------------------------------------------------------------

def answer_baseline(question: str, model: str = OLLAMA_MODEL) -> str:
    prompt = f"Answer the following question as best you can:\n\n{question}"
    return ask_local_llm(prompt, model=model)


# ---------------------------------------------------------------------------
# 6) Pretty print
# ---------------------------------------------------------------------------

def pretty_print(result: dict) -> None:
    if result.get("error"):
        print(f"\n[Error] {result['error']}")
    print(f"\n[SPARQL Query]{' (repaired x'+str(result['repairs'])+')' if result['repaired'] else ''}")
    print(result["query"])
    rows = result.get("rows", [])
    vars_ = result.get("vars", [])
    if not rows:
        print("\n[No results returned]")
        return
    print(f"\n[Results] ({len(rows)} rows)")
    print("  " + " | ".join(vars_))
    print("  " + "-" * 60)
    for r in rows[:20]:
        print("  " + " | ".join(r))
    if len(rows) > 20:
        print(f"  ... ({len(rows)} total)")


# ---------------------------------------------------------------------------
# 7) CLI demo
# ---------------------------------------------------------------------------

def cli_demo() -> None:
    model = check_ollama()
    g = load_graph()
    schema = build_schema_summary(g)
    print(f"\n[Schema summary built — {len(schema)} chars]")
    print("\nAI Research KG RAG Demo — type 'quit' to exit\n")

    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue

        print("\n--- Baseline (no KG) ---")
        print(answer_baseline(q, model=model))

        print("\n--- RAG (NL->SPARQL->rdflib) ---")
        result = answer_with_rag(g, schema, q, model=model)
        pretty_print(result)
        print()


if __name__ == "__main__":
    cli_demo()
