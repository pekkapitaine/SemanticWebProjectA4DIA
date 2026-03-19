"""
Information Extraction: Named Entity Recognition + Relation Extraction.

Reads crawler_output.jsonl, applies spaCy NER and dependency parsing,
and outputs extracted_knowledge.csv.

Output: data/extracted_knowledge.csv
"""

import csv
import json
import logging
from pathlib import Path

import spacy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_PATH = Path(__file__).parent.parent.parent / "data" / "crawler_output.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "extracted_knowledge.csv"

# Entity types we care about for the AI Research domain
RELEVANT_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "DATE"}

# Relation extraction: dependency roles that signal subject/object
SUBJECT_DEPS = {"nsubj", "nsubjpass"}
OBJECT_DEPS = {"dobj", "attr", "pobj", "oprd"}

# ---------------------------------------------------------------------------
# NLP model (loaded once)
# ---------------------------------------------------------------------------

def load_model() -> spacy.Language:
    """Load spaCy model; fall back to smaller model if transformer not available."""
    for model_name in ("en_core_web_trf", "en_core_web_lg", "en_core_web_sm"):
        try:
            nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
            return nlp
        except OSError:
            continue
    raise RuntimeError(
        "No spaCy English model found. Run: python -m spacy download en_core_web_trf"
    )


# ---------------------------------------------------------------------------
# Relation extraction helpers
# ---------------------------------------------------------------------------

def extract_relations(doc: spacy.tokens.Doc) -> list[dict]:
    """
    For each sentence, find (subject_entity, verb, object_entity) triples
    using dependency parsing.
    """
    relations = []
    ent_spans = {ent.start: ent for ent in doc.ents}

    for sent in doc.sents:
        for token in sent:
            if token.pos_ != "VERB":
                continue

            subjects = [
                child for child in token.children if child.dep_ in SUBJECT_DEPS
            ]
            objects = [
                child for child in token.children if child.dep_ in OBJECT_DEPS
            ]

            for subj in subjects:
                for obj in objects:
                    subj_ent = _find_ent_for_token(subj, doc)
                    obj_ent = _find_ent_for_token(obj, doc)
                    if subj_ent and obj_ent and subj_ent.label_ in RELEVANT_LABELS:
                        relations.append(
                            {
                                "subject": subj_ent.text,
                                "subject_type": subj_ent.label_,
                                "predicate": token.lemma_,
                                "object": obj_ent.text,
                                "object_type": obj_ent.label_,
                                "sentence": sent.text.strip(),
                            }
                        )
    return relations


def _find_ent_for_token(
    token: spacy.tokens.Token, doc: spacy.tokens.Doc
) -> spacy.tokens.Span | None:
    """Return the entity span that contains this token, if any."""
    for ent in doc.ents:
        if ent.start <= token.i < ent.end:
            return ent
    return None


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found: {input_path}\n"
            "Run src/crawl/crawler.py first."
        )

    nlp = load_model()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "url",
        "entity",
        "entity_type",
        "subject",
        "subject_type",
        "predicate",
        "object",
        "object_type",
        "sentence",
    ]

    total_entities = 0
    total_relations = 0

    with open(input_path, encoding="utf-8") as in_file, open(
        output_path, "w", newline="", encoding="utf-8"
    ) as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        for line in in_file:
            record = json.loads(line)
            url = record["url"]
            text = record["text"]

            logger.info(f"Processing: {url}")

            # Process in chunks to handle long texts
            chunks = _chunk_text(text, max_chars=100_000)
            for chunk in chunks:
                doc = nlp(chunk)

                # --- NER rows ---
                seen_entities: set[tuple] = set()
                for ent in doc.ents:
                    if ent.label_ not in RELEVANT_LABELS:
                        continue
                    key = (ent.text.strip(), ent.label_)
                    if key in seen_entities:
                        continue
                    seen_entities.add(key)
                    writer.writerow(
                        {
                            "url": url,
                            "entity": ent.text.strip(),
                            "entity_type": ent.label_,
                            "subject": "",
                            "subject_type": "",
                            "predicate": "",
                            "object": "",
                            "object_type": "",
                            "sentence": "",
                        }
                    )
                    total_entities += 1

                # --- Relation rows ---
                relations = extract_relations(doc)
                for rel in relations:
                    writer.writerow(
                        {
                            "url": url,
                            "entity": "",
                            "entity_type": "",
                            **rel,
                        }
                    )
                    total_relations += 1

    logger.info(
        f"Done. {total_entities} entities, {total_relations} relations → {output_path}"
    )


def _chunk_text(text: str, max_chars: int = 100_000) -> list[str]:
    """Split text into chunks at sentence boundaries (rough split by paragraphs)."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > max_chars:
            if current:
                chunks.append(current)
            current = para
        else:
            current += "\n\n" + para if current else para
    if current:
        chunks.append(current)
    return chunks


if __name__ == "__main__":
    extract()
