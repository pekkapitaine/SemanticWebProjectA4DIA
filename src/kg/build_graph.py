"""
Step 1 — Build the initial private RDF Knowledge Base.

Reads data/extracted_knowledge.csv (produced by src/ie/ner.py) and converts
entities + relations into RDF triples using our domain ontology.

Output:
    kg_artifacts/initial_kg.ttl   — initial private KB in Turtle format
"""

import csv
import re
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, XSD, URIRef
from rdflib.namespace import SKOS

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

KG = Namespace("http://semanticweb.esilv.fr/aikg/")
ONT = Namespace("http://semanticweb.esilv.fr/aikg/")

INPUT_PATH = Path(__file__).parent.parent.parent / "data" / "extracted_knowledge.csv"
ONTOLOGY_PATH = Path(__file__).parent.parent.parent / "kg_artifacts" / "ontology.ttl"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "kg_artifacts" / "initial_kg.ttl"

# ---------------------------------------------------------------------------
# Mapping spaCy NER labels → ontology classes
# ---------------------------------------------------------------------------

NER_TO_CLASS = {
    "PERSON": ONT.Researcher,
    "ORG": ONT.Organization,
    "PRODUCT": ONT.AIModel,
    "WORK_OF_ART": ONT.Publication,
    "EVENT": ONT.Benchmark,
    "GPE": ONT.Organization,   # locations treated as org context
    "DATE": None,              # dates → datatype property, not class
}

# Known AI model names → force :AIModel type
AI_MODEL_KEYWORDS = {
    "bert", "gpt", "llama", "falcon", "mistral", "mixtral", "gemma",
    "starcoder", "codellama", "vicuna", "alpaca", "bloom", "opt",
    "t5", "roberta", "distilbert", "electra", "xlnet", "deberta",
    "whisper", "stable diffusion", "dall-e", "clip", "glm",
}

# Known organization names
ORG_KEYWORDS = {
    "google", "microsoft", "openai", "meta", "deepmind", "hugging face",
    "huggingface", "nvidia", "amazon", "anthropic", "cohere", "ai21",
    "stanford", "mit", "berkeley", "oxford", "cambridge", "inria",
    "bigscience", "eleutherai",
}


def _slugify(text: str) -> str:
    """Convert a text label to a valid URI local name."""
    text = text.strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text[:80]  # cap length


def _detect_class(entity: str, ner_label: str) -> URIRef | None:
    """Determine the best ontology class for an entity."""
    lower = entity.lower()
    if any(kw in lower for kw in AI_MODEL_KEYWORDS):
        return ONT.AIModel
    if any(kw in lower for kw in ORG_KEYWORDS):
        return ONT.Organization
    return NER_TO_CLASS.get(ner_label)


def build_graph(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Graph:
    g = Graph()
    g.bind("", KG)
    g.bind("ont", ONT)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("skos", SKOS)

    # Load the ontology
    g.parse(str(ONTOLOGY_PATH), format="turtle")

    entity_uris: dict[str, URIRef] = {}  # label → URI
    triple_count = 0

    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Pass 1: entities
    for row in rows:
        entity = row["entity"].strip()
        ner_label = row["entity_type"].strip()
        url = row["url"].strip()

        if not entity or not ner_label:
            continue

        slug = _slugify(entity)
        if not slug:
            continue

        uri = KG[slug]
        entity_uris[entity] = uri

        ont_class = _detect_class(entity, ner_label)
        if ont_class:
            g.add((uri, RDF.type, ont_class))
            triple_count += 1

        g.add((uri, RDFS.label, Literal(entity, lang="en")))
        g.add((uri, KG.sourceURL, Literal(url, datatype=XSD.anyURI)))
        triple_count += 2

    # Pass 2: relations
    for row in rows:
        subj_label = row["subject"].strip()
        pred_label = row["predicate"].strip()
        obj_label = row["object"].strip()

        if not subj_label or not pred_label or not obj_label:
            continue

        # Subject URI
        if subj_label in entity_uris:
            subj_uri = entity_uris[subj_label]
        else:
            slug = _slugify(subj_label)
            if not slug:
                continue
            subj_uri = KG[slug]
            g.add((subj_uri, RDFS.label, Literal(subj_label, lang="en")))
            triple_count += 1

        # Object URI (may be a literal for non-entity objects)
        if obj_label in entity_uris:
            obj_uri = entity_uris[obj_label]
            g.add((subj_uri, KG[_slugify(pred_label)], obj_uri))
        else:
            g.add((subj_uri, KG[_slugify(pred_label)], Literal(obj_label, lang="en")))
        triple_count += 1

        # Add predicate label
        pred_uri = KG[_slugify(pred_label)]
        g.add((pred_uri, RDF.type, OWL.ObjectProperty))
        g.add((pred_uri, RDFS.label, Literal(pred_label, lang="en")))
        triple_count += 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(output_path), format="turtle")

    print(f"Initial KB: {triple_count} explicit triples, {len(entity_uris)} entities")
    print(f"Saved to {output_path}")
    return g


if __name__ == "__main__":
    build_graph()
