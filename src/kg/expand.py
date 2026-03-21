"""
Step 4 — KB Expansion via SPARQL on Wikidata.

Strategy (efficient batch approach):
  1. Batch 1-hop: all core QIDs in one VALUES query
  2. Broad predicate queries: key AI-domain predicates with high LIMITs
  3. 2-hop from top connected entities (one batched query)

~15 total SPARQL queries instead of 200+, completes in ~2 minutes.
Target: 50,000 – 200,000 triples

Output: kg_artifacts/expanded.nt
"""

import time
from pathlib import Path

import httpx
from rdflib import Graph, Namespace, OWL, URIRef, Literal

ALIGNMENT_PATH = Path(__file__).parent.parent.parent / "kg_artifacts" / "alignment.ttl"
INITIAL_KG_PATH = Path(__file__).parent.parent.parent / "kg_artifacts" / "initial_kg.ttl"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "kg_artifacts" / "expanded.nt"

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
DELAY = 3.0
HEADERS = {
    "User-Agent": "SemanticWebKGProject/1.0 (https://github.com/esilv; educational) python-httpx/0.27",
    "Accept": "application/sparql-results+json",
}

WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

# Predicates to skip (noisy/admin/media)
SKIP_PREDICATES = {
    "P18", "P154", "P214", "P217", "P373", "P625", "P856",
    "P1612", "P6104", "P6375", "P2397", "P345", "P3417",
    "P244", "P268", "P269", "P227", "P349", "P950",
}


def _sparql(query: str, client: httpx.Client, retries: int = 3) -> list[dict]:
    for attempt in range(retries):
        try:
            r = client.post(
                SPARQL_ENDPOINT,
                data={"query": query, "format": "json"},
                timeout=58,
            )
            r.raise_for_status()
            return r.json().get("results", {}).get("bindings", [])
        except Exception as e:
            print(f"  [WARN] attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(8)
    return []


def _ok_predicate(pred_uri: str) -> bool:
    if "prop/direct/" in pred_uri:
        pid = pred_uri.split("/")[-1]
        return pid not in SKIP_PREDICATES
    return True


def _add_bindings(g: Graph, bindings: list[dict], seen: set) -> int:
    added = 0
    for b in bindings:
        s_b = b.get("s", b.get("subject", {}))
        p_b = b.get("p", b.get("predicate", {}))
        o_b = b.get("o", b.get("object", {}))
        if s_b.get("type") != "uri" or p_b.get("type") != "uri":
            continue
        if not _ok_predicate(p_b["value"]):
            continue
        s = URIRef(s_b["value"])
        p = URIRef(p_b["value"])
        if o_b.get("type") == "uri":
            o = URIRef(o_b["value"])
        elif o_b.get("type") in ("literal", "typed-literal"):
            lang = o_b.get("xml:lang", "")
            if lang and lang != "en":
                continue  # keep only English literals
            o = Literal(o_b["value"])
        else:
            continue
        key = (str(s), str(p), str(o))
        if key not in seen:
            seen.add(key)
            g.add((s, p, o))
            added += 1
    return added


def expand(
    alignment_path: Path = ALIGNMENT_PATH,
    initial_kg_path: Path = INITIAL_KG_PATH,
    output_path: Path = OUTPUT_PATH,
    target: int = 80_000,
) -> None:
    # Load alignment to get core QIDs
    g_align = Graph()
    g_align.parse(str(alignment_path), format="turtle")
    core_qids = [
        str(o).split("/")[-1]
        for _, p, o in g_align.triples((None, OWL.sameAs, None))
        if str(o).startswith("http://www.wikidata.org/entity/Q")
    ]
    print(f"Core QIDs: {len(core_qids)}")

    g = Graph()
    g.parse(str(initial_kg_path), format="turtle")
    g.parse(str(alignment_path), format="turtle")

    seen: set = set()
    for s, p, o in g:
        seen.add((str(s), str(p), str(o)))
    print(f"Starting KB: {len(g)} triples")

    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:

        # ── Phase 1: Batch 1-hop from all core QIDs ──────────────────────────
        print("\n=== Phase 1: Batch 1-hop (all core QIDs) ===")
        batch_size = 40  # Wikidata VALUES clause works well up to ~50 items
        hop1_objects: set[str] = set()

        for i in range(0, len(core_qids), batch_size):
            batch = core_qids[i:i + batch_size]
            values = " ".join(f"wd:{q}" for q in batch)
            query = f"""
SELECT ?s ?p ?o WHERE {{
  VALUES ?s {{ {values} }}
  ?s ?p ?o .
  FILTER(!isLiteral(?o) || LANG(?o) = "en" || LANG(?o) = "")
}}
LIMIT 8000
"""
            bindings = _sparql(query, client)
            added = _add_bindings(g, bindings, seen)
            for b in bindings:
                o_b = b.get("o", {})
                if o_b.get("type") == "uri" and "wikidata.org/entity/Q" in o_b.get("value", ""):
                    hop1_objects.add(o_b["value"].split("/")[-1])
            print(f"  Batch {i//batch_size + 1}: +{added} triples (total: {len(g)})")
            time.sleep(DELAY)

        print(f"After Phase 1: {len(g)} triples | {len(hop1_objects)} hop-1 QIDs")

        # ── Phase 2: Broad predicate-controlled expansion ─────────────────────
        print("\n=== Phase 2: Broad predicate queries ===")
        broad_queries = [
            # (label, query)
            ("software instances", f"""
SELECT ?s ?p ?o WHERE {{
  ?s wdt:P31 ?o .
  ?s wdt:P178 [] .   # must have a developer
  BIND(wdt:P31 AS ?p)
}} LIMIT 25000"""),
            ("software developers", f"""
SELECT ?s ?p ?o WHERE {{
  ?s wdt:P178 ?o .
  BIND(wdt:P178 AS ?p)
}} LIMIT 30000"""),
            ("programming languages used", f"""
SELECT ?s ?p ?o WHERE {{
  ?s wdt:P277 ?o .
  BIND(wdt:P277 AS ?p)
}} LIMIT 20000"""),
            ("AI researchers employer", f"""
SELECT ?s ?p ?o WHERE {{
  ?s wdt:P108 ?o .
  ?s wdt:P101 wd:Q11660 .
  BIND(wdt:P108 AS ?p)
}} LIMIT 20000"""),
            ("AI researchers field", f"""
SELECT ?s ?p ?o WHERE {{
  ?s wdt:P101 ?o .
  ?s wdt:P101 wd:Q11660 .
  BIND(wdt:P101 AS ?p)
}} LIMIT 20000"""),
            ("notable works", f"""
SELECT ?s ?p ?o WHERE {{
  ?s wdt:P800 ?o .
  ?s wdt:P101 wd:Q11660 .
  BIND(wdt:P800 AS ?p)
}} LIMIT 15000"""),
            ("influenced by", f"""
SELECT ?s ?p ?o WHERE {{
  ?s wdt:P737 ?o .
  BIND(wdt:P737 AS ?p)
}} LIMIT 10000"""),
            ("part of", f"""
SELECT ?s ?p ?o WHERE {{
  ?s wdt:P361 ?o .
  ?s wdt:P31/wdt:P279* wd:Q7397 .
  BIND(wdt:P361 AS ?p)
}} LIMIT 10000"""),
            ("award received", f"""
SELECT ?s ?p ?o WHERE {{
  ?s wdt:P166 ?o .
  ?s wdt:P101 wd:Q11660 .
  BIND(wdt:P166 AS ?p)
}} LIMIT 10000"""),
            ("license", f"""
SELECT ?s ?p ?o WHERE {{
  ?s wdt:P275 ?o .
  ?s wdt:P178 [] .
  BIND(wdt:P275 AS ?p)
}} LIMIT 15000"""),
        ]

        for label, query in broad_queries:
            if len(g) >= target:
                print(f"  Target {target} reached, stopping.")
                break
            bindings = _sparql(query, client)
            added = _add_bindings(g, bindings, seen)
            print(f"  '{label}': +{added} triples (total: {len(g)})")
            time.sleep(DELAY)

        # ── Phase 3: 2-hop batch from top hop1 entities ───────────────────────
        if len(g) < target:
            print("\n=== Phase 3: 2-hop batch expansion ===")
            # Take first 80 hop1 QIDs
            hop1_sample = list(hop1_objects)[:80]
            for i in range(0, len(hop1_sample), batch_size):
                if len(g) >= target:
                    break
                batch = hop1_sample[i:i + batch_size]
                values = " ".join(f"wd:{q}" for q in batch)
                query = f"""
SELECT ?s ?p ?o WHERE {{
  VALUES ?s {{ {values} }}
  ?s ?p ?o .
  FILTER(!isLiteral(?o) || LANG(?o) = "en" || LANG(?o) = "")
}}
LIMIT 5000
"""
                bindings = _sparql(query, client)
                added = _add_bindings(g, bindings, seen)
                print(f"  2-hop batch {i//batch_size + 1}: +{added} (total: {len(g)})")
                time.sleep(DELAY)

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(output_path), format="nt")

    entities = set()
    predicates = set()
    for s, p, o in g:
        entities.add(str(s))
        if isinstance(o, URIRef):
            entities.add(str(o))
        predicates.add(str(p))

    print(f"\n=== Final KB ===")
    print(f"  Triples   : {len(g)}")
    print(f"  Entities  : {len(entities)}")
    print(f"  Relations : {len(predicates)}")
    print(f"  File      : {output_path}")


if __name__ == "__main__":
    expand()
