# Semantic Web Project — AI Research Knowledge Graph

> Course: Web Mining & Semantics — ESILV A4 DIA6

Full pipeline: Web Crawling → NER → RDF Knowledge Graph → Alignment → SWRL Reasoning → KGE → RAG

---

## Project Structure

```
project-root/
├─ src/
│  ├─ crawl/      # Lab 1 — Web crawler
│  ├─ ie/         # Lab 1 — NER & relation extraction
│  ├─ kg/         # Lab 2 — RDF graph + ontology + alignment
│  ├─ reason/     # Lab 3 — SWRL reasoning (OWLReady2)
│  ├─ kge/        # Lab 3 — Knowledge Graph Embeddings
│  └─ rag/        # Lab 4 — RAG pipeline (NL→SPARQL)
├─ data/
│  ├─ samples/    # Small sample files for reproducibility
│  └─ README.md
├─ kg_artifacts/
│  ├─ ontology.ttl
│  ├─ expanded.nt
│  └─ alignment.ttl
├─ reports/
│  └─ final_report.pdf
├─ notebooks/
├─ README.md
├─ requirements.txt
├─ .gitignore
└─ LICENSE
```

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

### Hardware Requirements

- Crawler + NER: any modern CPU, ~4 GB RAM
- KGE training: GPU recommended (NVIDIA, 6+ GB VRAM); CPU fallback supported
- RAG (Ollama): ~8 GB RAM minimum; GPU strongly recommended

---

## How to Run Each Module

### Lab 1 — Crawling

```bash
python src/crawl/crawler.py
# Output: data/crawler_output.jsonl
```

### Lab 1 — NER & Relation Extraction

```bash
python src/ie/ner.py
# Input:  data/crawler_output.jsonl
# Output: data/extracted_knowledge.csv
```

### Lab 2 — RDF Graph & Alignment

```bash
# Build initial RDF graph from NER output
python src/kg/build_graph.py
# Output: kg_artifacts/initial_kg.ttl  (~1369 triples, 348 entities)

# Entity linking + predicate alignment with Wikidata
python src/kg/align.py
# Output: kg_artifacts/alignment.ttl, data/entity_mapping.csv

# SPARQL expansion from Wikidata (~5 min)
python src/kg/expand.py
# Output: kg_artifacts/expanded.nt  (~85k triples)

# KB statistics
python src/kg/stats.py
# Output: kg_artifacts/kb_stats.txt
```

### Lab 3 — SWRL Reasoning

```bash
# Coming in Lab 3
python src/reason/reason.py
```

### Lab 3 — Knowledge Graph Embeddings

```bash
# Coming in Lab 3
python src/kge/train.py
python src/kge/evaluate.py
```

### Lab 4 — RAG Demo

```bash
# Requires Ollama running locally
ollama pull mistral
python src/rag/demo.py
```

---

## Ollama Setup

1. Download and install Ollama from https://ollama.com
2. Pull a model: `ollama pull mistral`
3. Ollama runs as a local server on port 11434 by default

---

## Data

Large data files are not committed. Run the crawler to regenerate:

```bash
python src/crawl/crawler.py
```

Sample files are available in `data/samples/`.

---

## Screenshot

*(to be added after RAG demo — Lab 4)*
