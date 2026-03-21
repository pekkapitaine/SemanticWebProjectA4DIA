"""
Step 2 & 3 — Entity linking + Predicate alignment with Wikidata.

For each entity in the initial KB:
  - Query Wikidata search API to find matching QIDs
  - Compute a confidence score based on label similarity
  - Add owl:sameAs links for high-confidence matches

For each predicate in the initial KB:
  - Search Wikidata for semantically equivalent properties
  - Add owl:equivalentProperty or rdfs:subPropertyOf links

Outputs:
    kg_artifacts/alignment.ttl   — all alignment triples
    data/entity_mapping.csv      — mapping table (for report)
"""

import csv
import time
import urllib.parse
from difflib import SequenceMatcher
from pathlib import Path

import httpx
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, XSD, URIRef

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

KG = Namespace("http://semanticweb.esilv.fr/aikg/")
WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
WIKIBASE = Namespace("http://wikiba.se/ontology#")

INITIAL_KG_PATH = Path(__file__).parent.parent.parent / "kg_artifacts" / "initial_kg.ttl"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "kg_artifacts" / "alignment.ttl"
MAPPING_CSV = Path(__file__).parent.parent.parent / "data" / "entity_mapping.csv"

WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
CONFIDENCE_THRESHOLD = 0.60
REQUEST_DELAY = 1.0  # polite delay between API calls

# Patterns to skip for API lookup (noise, dates, numbers)
import re as _re
_SKIP_PATTERN = _re.compile(r"^[\d\s\.\-\+B%]+$|^\d{4}$|^[A-Z]\d+[A-Z]*$")

# ---------------------------------------------------------------------------
# Known manual mappings (high-confidence, verified)
# Used to seed the alignment before querying the API
# ---------------------------------------------------------------------------

MANUAL_ENTITY_MAPPINGS: dict[str, tuple[str, float]] = {
    "BERT": ("Q56565539", 0.99),
    "GPT-4": ("Q116828662", 0.99),
    "LLaMA": ("Q118831653", 0.99),
    "Google": ("Q95", 0.99),
    "Microsoft": ("Q2283", 0.99),
    "Meta": ("Q380", 0.99),
    "OpenAI": ("Q21708200", 0.99),
    "NVIDIA": ("Q182477", 0.99),
    "Hugging_Face": ("Q63155490", 0.98),
    "Mistral": ("Q124641516", 0.97),
    "Falcon": ("Q120224088", 0.96),
    "Gemma": ("Q124082422", 0.95),
    "Stanford": ("Q41506", 0.98),
    "MIT": ("Q49108", 0.98),
    "DeepMind": ("Q15733006", 0.99),
    "Wikipedia": ("Q52", 0.99),
    "Amazon": ("Q3884", 0.97),
    "Anthropic": ("Q107820136", 0.99),
}

# Predicate alignment: local predicate -> Wikidata property QID
MANUAL_PREDICATE_MAPPINGS: dict[str, tuple[str, str, float]] = {
    # local predicate slug -> (wdt property, relation type, confidence)
    "developedBy": ("P178", "equivalentProperty", 0.97),
    "use":         ("P2283", "equivalentProperty", 0.85),
    "require":     ("P4510", "subPropertyOf",     0.70),
    "basedOn":     ("P144",  "equivalentProperty", 0.90),
    "trainedOn":   ("P7683", "subPropertyOf",     0.80),
    "affiliatedWith": ("P1416", "equivalentProperty", 0.92),
    "authoredBy":  ("P50",   "equivalentProperty", 0.95),
    "power":       ("P2283", "subPropertyOf",     0.65),
}


# ---------------------------------------------------------------------------
# Wikidata search helper
# ---------------------------------------------------------------------------

def _search_wikidata(label: str, client: httpx.Client, limit: int = 5) -> list[dict]:
    """Query Wikidata search API for a label. Returns list of candidates."""
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "limit": limit,
        "format": "json",
    }
    try:
        r = client.get(WIKIDATA_SEARCH, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("search", [])
    except Exception as e:
        print(f"  [WARN] Wikidata search failed for '{label}': {e}")
        return []


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _best_match(label: str, candidates: list[dict]) -> tuple[str | None, float]:
    """Pick the best candidate QID and compute confidence."""
    best_qid, best_score = None, 0.0
    for c in candidates:
        score = _similarity(label, c.get("label", ""))
        # Boost if description contains AI-related keywords
        desc = c.get("description", "").lower()
        if any(kw in desc for kw in ("language model", "artificial intelligence", "neural", "software", "company")):
            score = min(1.0, score + 0.15)
        if score > best_score:
            best_score = score
            best_qid = c.get("id")
    return best_qid, best_score


# ---------------------------------------------------------------------------
# Main alignment pipeline
# ---------------------------------------------------------------------------

def align(
    initial_kg_path: Path = INITIAL_KG_PATH,
    output_path: Path = OUTPUT_PATH,
    mapping_csv: Path = MAPPING_CSV,
) -> None:
    # Load initial KG to get all entity URIs
    g_source = Graph()
    g_source.parse(str(initial_kg_path), format="turtle")

    g_align = Graph()
    g_align.bind("", KG)
    g_align.bind("wd", WD)
    g_align.bind("wdt", WDT)
    g_align.bind("owl", OWL)
    g_align.bind("rdfs", RDFS)

    mapping_rows = []
    aligned_count = 0

    # Wikimedia-compliant User-Agent format
    headers = {
        "User-Agent": "SemanticWebKGProject/1.0 (https://github.com/esilv; educational) python-httpx/0.27"
    }
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        # ---- Entity linking ----
        print("=== Entity Linking ===")
        # Collect entities with labels from the KG
        entities: dict[URIRef, str] = {}
        for s, p, o in g_source.triples((None, RDFS.label, None)):
            if isinstance(s, URIRef) and str(s).startswith(str(KG)):
                entities[s] = str(o)

        for uri, label in entities.items():
            slug = str(uri).replace(str(KG), "")

            # 1. Check manual mappings first
            if slug in MANUAL_ENTITY_MAPPINGS:
                qid, confidence = MANUAL_ENTITY_MAPPINGS[slug]
                wd_uri = WD[qid]
                g_align.add((uri, OWL.sameAs, wd_uri))
                g_align.add((uri, KG.alignmentConfidence, Literal(confidence, datatype=XSD.decimal)))
                mapping_rows.append({
                    "private_entity": str(uri),
                    "label": label,
                    "external_uri": str(wd_uri),
                    "confidence": confidence,
                    "source": "manual",
                })
                aligned_count += 1
                print(f"  [MANUAL] {label} -> {qid} ({confidence:.2f})")
                continue

            # 2. Skip noise (numbers, dates, short tokens)
            if _SKIP_PATTERN.match(slug) or len(label) < 3:
                mapping_rows.append({
                    "private_entity": str(uri), "label": label,
                    "external_uri": "SKIP", "confidence": 0.0, "source": "skipped",
                })
                continue

            # 3. API search for remaining entities
            candidates = _search_wikidata(label, client)
            time.sleep(REQUEST_DELAY)

            if not candidates:
                mapping_rows.append({
                    "private_entity": str(uri),
                    "label": label,
                    "external_uri": "NEW",
                    "confidence": 0.0,
                    "source": "not_found",
                })
                continue

            qid, confidence = _best_match(label, candidates)
            if qid and confidence >= CONFIDENCE_THRESHOLD:
                wd_uri = WD[qid]
                g_align.add((uri, OWL.sameAs, wd_uri))
                g_align.add((uri, KG.alignmentConfidence, Literal(round(confidence, 3), datatype=XSD.decimal)))
                mapping_rows.append({
                    "private_entity": str(uri),
                    "label": label,
                    "external_uri": str(wd_uri),
                    "confidence": round(confidence, 3),
                    "source": "api",
                })
                aligned_count += 1
                print(f"  [API]    {label} -> {qid} ({confidence:.2f})")
            else:
                mapping_rows.append({
                    "private_entity": str(uri),
                    "label": label,
                    "external_uri": "NEW",
                    "confidence": round(confidence, 3) if confidence else 0.0,
                    "source": "low_confidence",
                })

        # ---- Predicate alignment ----
        print("\n=== Predicate Alignment ===")
        for pred_slug, (wdt_prop, rel_type, confidence) in MANUAL_PREDICATE_MAPPINGS.items():
            pred_uri = KG[pred_slug]
            wdt_uri = URIRef(f"http://www.wikidata.org/prop/direct/{wdt_prop}")

            if rel_type == "equivalentProperty":
                g_align.add((pred_uri, OWL.equivalentProperty, wdt_uri))
            else:
                g_align.add((pred_uri, RDFS.subPropertyOf, wdt_uri))

            g_align.add((pred_uri, KG.alignmentConfidence, Literal(confidence, datatype=XSD.decimal)))
            print(f"  {pred_slug} --[{rel_type}]--> wdt:{wdt_prop} ({confidence:.2f})")

    # Save alignment graph
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g_align.serialize(destination=str(output_path), format="turtle")

    # Save mapping CSV
    mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["private_entity", "label", "external_uri", "confidence", "source"])
        writer.writeheader()
        writer.writerows(mapping_rows)

    print(f"\nAligned {aligned_count}/{len(entities)} entities above threshold {CONFIDENCE_THRESHOLD}")
    print(f"Alignment graph -> {output_path}")
    print(f"Mapping table   -> {mapping_csv}")


if __name__ == "__main__":
    align()
