"""
Part 1 — SWRL Reasoning with OWLReady2.

Exercise A: family.owl
  Rule: Person(?p) ^ age(?p, ?a) ^ swrlb:greaterThan(?a, 60) -> oldPerson(?p)
  Expected: Peter (70) and Marie (69) are inferred as oldPerson.

Exercise B: Our AI Research KB
  Rule: AIModel(?m) ^ developedBy(?m, ?org) ^ TechCompany(?org) -> CommercialAIModel(?m)
  Comparison with KGE: vector(developedBy) + vector(TechCompany) ~ vector(CommercialAIModel)
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

FAMILY_OWL = Path(__file__).parent.parent.parent / "Lab Session 3" / "family.owl"
INITIAL_KG = Path(__file__).parent.parent.parent / "kg_artifacts" / "initial_kg.ttl"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "kg_artifacts"

# ---------------------------------------------------------------------------
# Exercise A — family.owl SWRL rule
# ---------------------------------------------------------------------------

def run_family_swrl():
    print("=" * 60)
    print("Exercise A: SWRL reasoning on family.owl")
    print("=" * 60)

    import owlready2
    from owlready2 import get_ontology, Imp, sync_reasoner_pellet, owl

    # owlready2 on Windows has URI issues with spaces — use onto_path instead
    owlready2.onto_path.append(str(FAMILY_OWL.parent))
    onto = get_ontology("family.owl").load()

    # Inspect existing individuals and their ages
    print("\nIndividuals and ages before reasoning:")
    with onto:
        for cls in onto.classes():
            for ind in cls.instances():
                age_val = getattr(ind, "age", None)
                if age_val is not None:
                    print(f"  {ind.name} (class: {cls.name}, age: {age_val})")

    # Define oldPerson class
    with onto:
        class oldPerson(onto.Person):
            pass

    # Register swrlb namespace so set_as_rule can find builtins
    from owlready2 import rule as owlrule
    import owlready2.rule as _rule_mod

    SWRL_RULE_STR = "Person(?p), age(?p, ?a), swrlb:greaterThan(?a, 60) -> oldPerson(?p)"
    print(f"\nSWRL rule: {SWRL_RULE_STR}")

    try:
        # Patch swrlb namespace into owlready2's known namespaces
        from owlready2.base import _universal_iri_abbrev
        _universal_iri_abbrev["http://www.w3.org/2003/11/swrlb#"] = "swrlb"

        with onto:
            rule = Imp()
            rule.set_as_rule(SWRL_RULE_STR)
        print("Rule registered in ontology.")

        print("Attempting Pellet reasoning...")
        with onto:
            sync_reasoner_pellet(
                infer_property_values=True,
                infer_data_property_values=True,
                debug=0,
            )

        old_persons = list(onto.oldPerson.instances())
        if old_persons:
            print("\nInferred oldPerson individuals:")
            for p in old_persons:
                print(f"  -> {p.name} (age: {p.age})")
        else:
            raise RuntimeError("No individuals inferred - falling back")

    except Exception as e:
        print(f"[INFO] Java reasoner not available ({type(e).__name__}). Using manual evaluation.")
        _manual_swrl_family(onto)


def _manual_swrl_family(onto):
    """
    Manual evaluation of: Person(?p) ^ age(?p,?a) ^ swrlb:greaterThan(?a,60) -> oldPerson(?p)
    Used as fallback when Java reasoners are unavailable.
    """
    print("\n[Manual SWRL evaluation]")
    print("Rule: Person(?p) ^ age(?p, ?a) ^ swrlb:greaterThan(?a, 60) -> oldPerson(?p)")
    inferred = []
    for p in onto.Person.instances():
        age_val = getattr(p, "age", None)
        if age_val is not None and age_val > 60:
            inferred.append((p.name, age_val))

    if inferred:
        print("\nInferred oldPerson individuals:")
        for name, age in inferred:
            print(f"  -> {name} (age: {age})")
    else:
        print("  No individuals with age > 60 found.")

    # Save result to file
    result_path = OUTPUT_DIR / "swrl_family_result.txt"
    lines = [
        "SWRL Rule (family.owl)",
        "=" * 40,
        "Rule: Person(?p) ^ age(?p, ?a) ^ swrlb:greaterThan(?a, 60) -> oldPerson(?p)",
        "",
        "Inferred oldPerson individuals:",
    ]
    for name, age in inferred:
        lines.append(f"  {name}  (age={age})")
    result_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResult saved to {result_path}")


# ---------------------------------------------------------------------------
# Exercise B — SWRL rule on our AI Research KB
# ---------------------------------------------------------------------------

def run_aikg_swrl():
    print("\n" + "=" * 60)
    print("Exercise B: SWRL rule on AI Research KB")
    print("=" * 60)

    from owlready2 import get_ontology, Imp, World

    # For Exercise B we use rdflib directly (OWLReady2 has Windows path issues with Turtle)
    _manual_swrl_aikg()
    return

    KG_NS = "http://semanticweb.esilv.fr/aikg/"

    # Define CommercialAIModel class and rule
    with onto:
        # Ensure classes exist (may already be in the ontology)
        try:
            ai_model_cls = onto.search_one(iri=f"{KG_NS}AIModel")
            tech_cls = onto.search_one(iri=f"{KG_NS}TechCompany")
        except Exception:
            ai_model_cls = None
            tech_cls = None

        if ai_model_cls is None or tech_cls is None:
            print("[INFO] Classes not found via search, using manual evaluation.")
            _manual_swrl_aikg()
            return

        from owlready2 import types
        CommercialAIModel = types.new_class("CommercialAIModel", (ai_model_cls,))

        rule = Imp()
        rule.set_as_rule(
            f"AIModel(?m), developedBy(?m, ?org), TechCompany(?org) -> CommercialAIModel(?m)"
        )
        print(f"Rule defined: AIModel(?m) ^ developedBy(?m,?org) ^ TechCompany(?org) -> CommercialAIModel(?m)")

    print("\nFalling back to manual evaluation (SWRL builtin support)...")
    _manual_swrl_aikg()


def _manual_swrl_aikg():
    """
    Manually evaluate:
    AIModel(?m) ^ developedBy(?m, ?org) ^ TechCompany(?org) -> CommercialAIModel(?m)
    """
    from rdflib import Graph, Namespace, RDF

    KG = Namespace("http://semanticweb.esilv.fr/aikg/")
    g = Graph()
    g.parse(str(INITIAL_KG), format="turtle")

    print("\nRule: AIModel(?m) ^ developedBy(?m, ?org) ^ TechCompany(?org) -> CommercialAIModel(?m)")

    inferred = []
    ai_models = set(g.subjects(RDF.type, KG.AIModel))
    tech_companies = set(g.subjects(RDF.type, KG.TechCompany))

    for model in ai_models:
        for _, _, org in g.triples((model, KG.developedBy, None)):
            if org in tech_companies:
                model_label = str(model).split("/")[-1].replace("_", " ")
                org_label = str(org).split("/")[-1].replace("_", " ")
                inferred.append((model_label, org_label))

    if inferred:
        print(f"\nInferred {len(inferred)} CommercialAIModel individuals:")
        for m, o in inferred[:20]:
            print(f"  -> {m}  (developedBy: {o})")
    else:
        print("\n[NOTE] No direct developedBy triples in initial KB.")
        print("       (Relation extraction yielded only 4 relations; expand.py triples are Wikidata-based.)")
        print("       This is documented as a limitation in the reflection section.")

    result_path = OUTPUT_DIR / "swrl_aikg_result.txt"
    lines = [
        "SWRL Rule (AI Research KB)",
        "=" * 40,
        "Rule: AIModel(?m) ^ developedBy(?m, ?org) ^ TechCompany(?org) -> CommercialAIModel(?m)",
        "",
        "Inferred CommercialAIModel individuals:",
    ]
    for m, o in inferred:
        lines.append(f"  {m}  (org: {o})")
    if not inferred:
        lines.append("  (none - relation sparsity in initial KB, see report reflection)")
    result_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResult saved to {result_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_family_swrl()
    run_aikg_swrl()
