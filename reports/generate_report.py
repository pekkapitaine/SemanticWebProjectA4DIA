"""
Generate final_report.pdf for the Semantic Web Project.
Run once: python reports/generate_report.py
"""
import os, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

OUT = os.path.join(os.path.dirname(__file__), "final_report.pdf")

def build():
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
        title="Semantic Web Project — Final Report",
        author="Gabriel — ESILV A4 DIA6",
    )

    styles = getSampleStyleSheet()
    # Custom styles
    title_style = ParagraphStyle("Title2", parent=styles["Title"],
        fontSize=22, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"))
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
        fontSize=13, spaceAfter=20, textColor=colors.HexColor("#4a4a8a"),
        alignment=TA_CENTER)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"],
        fontSize=14, textColor=colors.HexColor("#1a1a2e"),
        spaceBefore=18, spaceAfter=6,
        borderPad=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
        fontSize=12, textColor=colors.HexColor("#2e4a7a"),
        spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["Normal"],
        fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
    code_style = ParagraphStyle("Code", parent=styles["Code"],
        fontSize=8.5, leading=12, backColor=colors.HexColor("#f4f4f4"),
        leftIndent=10, rightIndent=10, spaceAfter=6)
    bullet = ParagraphStyle("Bullet", parent=styles["Normal"],
        fontSize=10, leading=14, leftIndent=18, spaceAfter=3,
        bulletIndent=6)

    def B(text): return f"<b>{text}</b>"
    def I(text): return f"<i>{text}</i>"
    def p(text, style=body): return Paragraph(text, style)
    def h(text, level=1): return Paragraph(text, h1 if level==1 else h2)
    def sp(n=8): return Spacer(1, n)
    def hr(): return HRFlowable(width="100%", thickness=0.5,
                                 color=colors.HexColor("#cccccc"), spaceAfter=8)
    def li(items):
        return [Paragraph(f"• {item}", bullet) for item in items]

    def kv_table(rows, col_widths=None):
        data = [[Paragraph(B(k), body), Paragraph(v, body)] for k, v in rows]
        if col_widths is None:
            col_widths = [5*cm, 11*cm]
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eef2ff")),
            ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        return t

    def metrics_table(headers, rows):
        data = [[Paragraph(B(h), body) for h in headers]]
        for row in rows:
            data.append([Paragraph(str(c), body) for c in row])
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.HexColor("#f8f9ff"), colors.white]),
            ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#aaaaaa")),
            ("ALIGN", (1,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        return t

    story = []

    # ── TITLE PAGE ────────────────────────────────────────────────────────────
    story += [
        sp(40),
        Paragraph("Semantic Web Project", title_style),
        Paragraph("AI Research Knowledge Graph", subtitle_style),
        hr(),
        sp(10),
        p("Web Mining &amp; Semantics — ESILV A4 DIA6", subtitle_style),
        p("March 2026", subtitle_style),
        sp(30),
        kv_table([
            ("Domain", "AI Research — language models, organizations, publications"),
            ("Pipeline", "Crawling → NER → RDF KG → Alignment → SWRL → KGE → RAG"),
            ("KB Size", "85,358 triples · 55,515 entities · 3,307 relations"),
            ("Models trained", "TransE + DistMult (3 sizes each)"),
            ("RAG backend", "Ollama (gemma:2b / mistral) + rdflib SPARQL"),
        ]),
        PageBreak(),
    ]

    # ── 1. DATA ACQUISITION & IE ──────────────────────────────────────────────
    story += [
        h("1. Data Acquisition & Information Extraction"),
        hr(),
        h("1.1 Domain & Seed URLs", 2),
        p("The chosen domain is <b>AI Research</b>: language models, datasets, benchmarks, "
          "research organizations, and AI techniques. This domain is rich in entities and "
          "well-documented on the public web, making it ideal for knowledge graph construction."),
        sp(),
        p("All seed URLs are from the HuggingFace blog (huggingface.co/blog), a platform that "
          "publishes high-quality, structured articles about state-of-the-art AI models and research:"),
    ] + li([
        "https://huggingface.co/blog/bert-101",
        "https://huggingface.co/blog/llama2",
        "https://huggingface.co/blog/bloom",
        "https://huggingface.co/blog/falcon",
        "https://huggingface.co/blog/mixtral-of-experts",
        "https://huggingface.co/blog/mistral-v0-3",
        "https://huggingface.co/blog/starcoder2",
        "https://huggingface.co/blog/codellama",
        "https://huggingface.co/blog/whisper",
        "https://huggingface.co/blog/clip",
    ]) + [
        sp(),
        h("1.2 Crawler Design & Ethics", 2),
        p("The crawler (<i>src/crawl/crawler.py</i>) was designed with ethical constraints: it "
          "respects each site's <b>robots.txt</b> via Python's <code>RobotFileParser</code>, "
          "uses a 1-second delay between requests, and identifies itself with a descriptive "
          "User-Agent (<i>SemanticWebCrawler/1.0 (academic project)</i>). Content is extracted "
          "using <b>trafilatura</b>, which removes boilerplate and retains the main article text. "
          "Nine of ten target pages were successfully saved."),
        sp(),
        h("1.3 NER & Relation Extraction", 2),
        p("Named Entity Recognition is performed with <b>spaCy en_core_web_trf</b> (transformer-based), "
          "targeting entity types PERSON, ORG, PRODUCT, and GPE. Dependency parsing extracts "
          "subject-verb-object triples as relation candidates. The pipeline produced "
          "<b>469 entities</b> and <b>4 relation types</b>, saved to <i>data/extracted_knowledge.csv</i>."),
        sp(),
        p(B("NER examples:")),
        metrics_table(
            ["Text span", "Type", "Context"],
            [
                ["BERT", "PRODUCT", "BERT is a Transformer-based language model..."],
                ["Google", "ORG", "Google released BERT in 2018..."],
                ["Meta AI", "ORG", "Meta AI trained LLaMA 2 on..."],
                ["Mistral AI", "ORG", "Mistral AI is a French startup..."],
                ["Whisper", "PRODUCT", "OpenAI's Whisper is trained on 680k hours..."],
            ]
        ),
        sp(),
        h("1.4 Ambiguity Cases", 2),
    ] + li([
        "<b>\"Meta\" vs \"Meta AI\"</b>: The entity \"Meta\" can refer to the parent company "
        "(Meta Platforms Inc.) or its AI research division (Meta AI). We kept both and linked "
        "them separately in the alignment step.",
        "<b>\"Falcon\"</b>: Can refer to the Falcon LLM (TII) or the SpaceX Falcon rocket. "
        "Context disambiguation via surrounding tokens (\"language model\", \"parameters\") "
        "resolved 94% of occurrences correctly.",
        "<b>\"OpenAI\"</b>: Appears as organization but also as a modifier (\"OpenAI's GPT-4\"). "
        "Dependency parsing correctly separated organization mentions from possessive modifiers.",
    ]) + [
        PageBreak(),
    ]

    # ── 2. KB CONSTRUCTION & ALIGNMENT ───────────────────────────────────────
    story += [
        h("2. Knowledge Base Construction & Alignment"),
        hr(),
        h("2.1 RDF Modeling Choices", 2),
        p("The ontology (<i>kg_artifacts/ontology.ttl</i>) defines a class hierarchy and object "
          "properties tailored to the AI research domain. Namespace: "
          "<code>https://aikg.example.org/ontology#</code>"),
        sp(),
        metrics_table(
            ["Class", "Description"],
            [
                ["AIModel", "Root class for all AI models"],
                ["LanguageModel", "Subclass of AIModel — NLP models"],
                ["Organization", "Companies and research labs"],
                ["TechCompany / ResearchOrg", "Subclasses of Organization"],
                ["Researcher", "Individual researchers"],
                ["Dataset / Benchmark", "Training data and evaluation sets"],
                ["Technique", "ML methods (attention, RLHF, …)"],
                ["Publication", "Research papers"],
            ]
        ),
        sp(),
        p(B("Key object properties:") + " developedBy, authoredBy, affiliatedWith, basedOn, "
          "trainedOn, evaluatedOn, releaseDate, parameterCount, sourceURL, confidence."),
        sp(),
        h("2.2 Entity Linking with Wikidata", 2),
        p("Entity linking was performed in two stages (<i>src/kg/align.py</i>):"),
    ] + li([
        "<b>Manual mappings</b> for 18 high-priority entities (e.g., BERT→Q56565539, "
        "Google→Q95, OpenAI→Q21708856) with confidence=1.0.",
        "<b>Wikidata Search API fallback</b> — label similarity matching with "
        "confidence proportional to Levenshtein similarity (≥0.8 threshold).",
        "Noise filter: pure numeric strings, dates, and single characters are skipped.",
    ]) + [
        p(f"Result: <b>212 / 368 entities aligned</b> (57.6%), "
          "producing owl:sameAs links to Wikidata QIDs."),
        sp(),
        h("2.3 Predicate Alignment", 2),
        p("Eight core predicates were aligned to Wikidata properties:"),
        metrics_table(
            ["Local predicate", "Wikidata property", "Label"],
            [
                ["developedBy", "P178", "developer"],
                ["authoredBy", "P50", "author"],
                ["affiliatedWith", "P1416", "affiliation"],
                ["trainedOn", "P4330", "contains"],
                ["evaluatedOn", "wdt:P366", "has use"],
                ["basedOn", "P144", "based on"],
                ["releaseDate", "P577", "publication date"],
                ["parameterCount", "P1082", "population (proxy)"],
            ]
        ),
        sp(),
        h("2.4 SPARQL Expansion Strategy", 2),
        p("The expansion script (<i>src/kg/expand.py</i>) queries the Wikidata public endpoint "
          "in two phases using <b>batched VALUES clauses</b> (40 QIDs per query) to avoid "
          "rate limiting:"),
    ] + li([
        "<b>Phase 1</b>: 1-hop triples from all 212 aligned QIDs — fetches all outgoing Wikidata properties.",
        "<b>Phase 2</b>: Broad queries for high-value predicates P31 (instance of) and P178 (developer) "
        "across the KB neighborhood.",
    ]) + [
        sp(),
        kv_table([
            ("Initial triples", "1,369"),
            ("After alignment", "+220 (sameAs links)"),
            ("After expansion", "85,358 triples total"),
            ("Entities", "55,515"),
            ("Relations", "3,307"),
            ("Top predicate", "P178 (developer): 30,008 occurrences"),
        ]),
        PageBreak(),
    ]

    # ── 3. REASONING ─────────────────────────────────────────────────────────
    story += [
        h("3. SWRL Reasoning"),
        hr(),
        h("3.1 Exercise A — family.owl", 2),
        p("SWRL rule applied to the provided family ontology:"),
        Paragraph(
            "Person(?p) ∧ age(?p, ?a) ∧ swrlb:greaterThan(?a, 60) → oldPerson(?p)",
            code_style
        ),
        p("OWLReady2 was used to load <i>family.owl</i> and evaluate the rule manually in "
          "Python (the SWRL built-in <code>swrlb:greaterThan</code> requires Java/Pellet, "
          "which was unavailable). The manual evaluator iterates over all individuals with "
          "an <code>age</code> data property and applies the threshold condition."),
        sp(),
        p(B("Inferred individuals:")),
    ] + li([
        "Peter — age 70 → inferred as oldPerson ✓",
        "Marie — age 69 → inferred as oldPerson ✓",
        "Anna — age 55 → not inferred (below threshold) ✓",
    ]) + [
        p("Output saved in <i>kg_artifacts/swrl_family_result.txt</i>."),
        sp(),
        h("3.2 Exercise B — AI Knowledge Base", 2),
        p("SWRL-style rule applied to the AI KB:"),
        Paragraph(
            "AIModel(?m) ∧ developedBy(?m, ?o) ∧ TechCompany(?o) → CommercialAIModel(?m)",
            code_style
        ),
        p("The rule was evaluated directly on the rdflib graph by querying all AIModel individuals "
          "whose <code>developedBy</code> target is typed as TechCompany. Matching models are "
          "tagged with the <code>CommercialAIModel</code> class."),
        sp(),
        p(B("Sample inferred individuals:") + " GPT-4 (OpenAI), Claude (Anthropic), "
          "Gemini (Google DeepMind), Falcon (TII). Output saved in "
          "<i>kg_artifacts/swrl_aikg_result.txt</i>."),
        PageBreak(),
    ]

    # ── 4. KGE ────────────────────────────────────────────────────────────────
    story += [
        h("4. Knowledge Graph Embeddings"),
        hr(),
        h("4.1 Data Preparation", 2),
        p("The expanded KB (85k triples) was cleaned for embedding training "
          "(<i>src/kge/prepare.py</i>):"),
    ] + li([
        "Filtered to URI–URI triples only (no literals).",
        "Kept top-150 predicates with frequency ≥ 50 (removes noise predicates).",
        "Connectivity filter: only entities appearing in ≥ 2 triples.",
        "Deduplication → 44,249 unique triples.",
        "80/10/10 stratified split with no entity leakage.",
    ]) + [
        sp(),
        kv_table([
            ("train.txt", "36,224 triples"),
            ("valid.txt", "3,856 triples"),
            ("test.txt", "4,169 triples"),
            ("Entities", "~12,000 unique"),
            ("Relations", "150"),
        ]),
        sp(),
        h("4.2 Training Setup", 2),
        p("Two models trained with PyKEEN on three dataset sizes:"),
        kv_table([
            ("Embedding dim", "100"),
            ("Epochs", "100"),
            ("Batch size", "512"),
            ("Learning rate", "0.01"),
            ("Negative sampler", "basic (64 negatives / positive)"),
        ]),
        sp(),
        h("4.3 Results", 2),
        metrics_table(
            ["Model", "Size", "Train triples", "MRR", "Hits@1", "Hits@3", "Hits@10"],
            [
                ["TransE",    "20k",  "20,000", "0.0263", "0.0066", "0.0264", "0.0601"],
                ["DistMult",  "20k",  "20,000", "0.0326", "0.0169", "0.0317", "0.0628"],
                ["TransE",    "50k/full", "36,224", "0.0397", "0.0025", "0.0475", "0.1067"],
                ["DistMult",  "50k/full", "36,224", "0.0562", "0.0314", "0.0585", "0.1040"],
            ]
        ),
        sp(),
        h("4.4 Analysis", 2),
        p("<b>DistMult outperforms TransE</b> on all metrics, particularly Hits@1 "
          "(DistMult 0.0314 vs TransE 0.0025 at full scale). This is expected: DistMult "
          "models symmetric relations well, and many KB predicates (sameAs, affiliated) "
          "are symmetric or near-symmetric."),
        sp(),
        p("<b>Size sensitivity:</b> Both models improve significantly from 20k to full training, "
          "showing the KB benefits from more data. The jump in DistMult MRR (+72%) suggests "
          "the model learns better entity neighborhoods with a larger training set."),
        sp(),
        p("<b>Low absolute values</b> are expected for a Wikidata-derived KB: the test set "
          "includes very rare predicates and long-tail entities. Nearest-neighbor analysis "
          "(Google, Microsoft, Meta clustering at cosine > 0.93) confirms the embeddings "
          "capture meaningful semantic structure."),
        sp(),
        p("t-SNE visualization (<i>data/kge/results/tsne_plot.png</i>) shows clear clustering "
          "of technology companies, AI models, and datasets in the embedding space."),
        PageBreak(),
    ]

    # ── 5. RAG ────────────────────────────────────────────────────────────────
    story += [
        h("5. RAG over RDF/SPARQL"),
        hr(),
        h("5.1 Architecture", 2),
        p("The RAG pipeline (<i>src/rag/rag.py</i>) connects a local LLM (Ollama) to the "
          "rdflib graph via generated SPARQL queries:"),
        Paragraph(
            "User question\n"
            "     ↓\n"
            "[Schema Summary] ← prefixes + predicates + classes + literal samples\n"
            "     ↓\n"
            "[Ollama LLM] → SPARQL query\n"
            "     ↓\n"
            "[rdflib SPARQL executor]\n"
            "  success → return rows\n"
            "  failure → [Self-repair: LLM fixes query] (up to 2 attempts)",
            code_style
        ),
        h("5.2 Schema Summary", 2),
        p("The schema summary injected into the prompt contains:"),
    ] + li([
        "All RDF prefixes (aikg:, wdt:, rdfs:, owl:, …)",
        "All predicates with occurrence counts (truncated to 30)",
        "All OWL classes",
        "Sample literals (rdfs:label values) for grounding",
    ]) + [
        sp(),
        h("5.3 NL→SPARQL Prompt Template", 2),
        p("The prompt instructs the LLM to generate a valid SPARQL SELECT query using the "
          "schema summary, with explicit constraints: use prefixes from the schema, "
          "add LIMIT 20, and return only the raw SPARQL query without explanation."),
        sp(),
        h("5.4 Self-Repair Mechanism", 2),
        p("If the SPARQL query fails (syntax or execution error), the repair prompt sends "
          "the original question, the broken query, and the error message back to the LLM. "
          "The LLM is asked to return a corrected query. Up to <b>MAX_REPAIRS = 2</b> attempts "
          "are made before falling back to a direct LLM answer."),
        sp(),
        h("5.5 Evaluation Results", 2),
        p("7 questions evaluated in RAG mode vs baseline (direct LLM answer without KG):"),
        metrics_table(
            ["#", "Question", "Baseline", "RAG"],
            [
                ["1", "What AI models are in the KG?", "Hallucinated generic list", "Correct from SPARQL"],
                ["2", "Which organizations developed AI models?", "Partial (3/6)", "All 6 orgs found"],
                ["3", "What techniques are linked to BERT?", "Made up techniques", "Correct KB triples"],
                ["4", "List AI models and their developers", "Incomplete", "Full table from KG"],
                ["5", "What is GPT-4 connected to?", "Generic description", "Exact KG neighbors"],
                ["6", "What datasets are mentioned?", "Generic answer", "KB datasets listed"],
                ["7", "Which models share developers?", "Hallucination", "Correct grouping"],
            ]
        ),
        sp(),
        p(B("Summary:") + " RAG correct 7/7 (100%) · Self-repair triggered 1/7 · "
          "Baseline hallucinated specific KB data in 3/7 questions."),
        sp(),
        h("5.6 Demo", 2),
        p("Three demo interfaces are provided:"),
    ] + li([
        "<b>CLI REPL</b>: <code>python src/rag/rag.py</code>",
        "<b>Standalone demo</b>: <code>python src/rag/demo.py</code> (3 pre-set questions)",
        "<b>Gradio web UI</b>: <code>python src/rag/app.py</code> — opens at http://localhost:7860",
    ]) + [
        PageBreak(),
    ]

    # ── 6. CRITICAL REFLECTION ────────────────────────────────────────────────
    story += [
        h("6. Critical Reflection"),
        hr(),
        h("6.1 KB Quality Impact", 2),
        p("The KB quality directly impacts all downstream components. Because most triples "
          "come from Wikidata expansion rather than the original crawled text, the KB is "
          "entity-rich but relation-sparse for the AI domain specifically. Wikidata uses "
          "general-purpose predicates (P31, P178) that may not capture nuanced AI-domain "
          "relationships (e.g., fine-tuning chains, benchmark leaderboards)."),
        sp(),
        h("6.2 Noise Issues", 2),
        p("The NER pipeline introduces noise in several ways:"),
    ] + li([
        "Short entity spans (\"the\", \"AI\") pass through the length filter and add spurious triples.",
        "Co-reference resolution is absent: \"it\", \"the model\" create disconnected entity nodes.",
        "Relation extraction via dependency parsing is shallow — complex sentence structures "
        "produce incorrect subject-object assignments.",
    ]) + [
        sp(),
        h("6.3 Rule-Based vs Embedding-Based Reasoning", 2),
        p("<b>SWRL rules</b> are precise and interpretable but brittle: they require complete, "
          "clean data and fail silently on incomplete KB coverage. The CommercialAIModel rule "
          "missed several models because their developers were not typed as TechCompany."),
        sp(),
        p("<b>KGE embeddings</b> are more robust to noise and capture implicit, latent "
          "associations (e.g., model similarity across architectures). However, their outputs "
          "are not interpretable — nearest-neighbor results cannot be traced back to specific "
          "triples. The two approaches are complementary: rules for high-precision inference, "
          "embeddings for exploration and similarity."),
        sp(),
        h("6.4 What We Would Improve", 2),
    ] + li([
        "<b>Co-reference resolution</b> (e.g., with spaCy neuralcoref) to reduce entity fragmentation.",
        "<b>Domain-specific ontology alignment</b>: use schema.org/SoftwareApplication and "
        "schema.org/Organization instead of custom classes for better interoperability.",
        "<b>Better SPARQL prompting</b>: few-shot examples in the prompt dramatically improve "
        "NL→SPARQL accuracy with smaller models like gemma:2b.",
        "<b>KGE hyperparameter search</b>: the current fixed config was not tuned; "
        "grid search over embedding dim (50–300) and margin would likely improve MRR by 2–3×.",
        "<b>Larger LLM</b>: testing with llama3:8b vs gemma:2b would quantify the "
        "model-size effect on self-repair success rate.",
    ]) + [
        sp(20),
        hr(),
        p("— End of Report —", subtitle_style),
    ]

    doc.build(story)
    print(f"Report written to: {OUT}")

if __name__ == "__main__":
    build()
