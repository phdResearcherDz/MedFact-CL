"""Entity normalization f_norm (UMLS via ScispaCy) and triple linearization."""
from typing import Dict, List, Optional


class UMLSNormalizer:
    """Links a mention to its top-ranked UMLS concept. Predicates are not normalized."""

    def __init__(self, spacy_model: str = "en_core_sci_sm"):
        import spacy
        from scispacy.linking import EntityLinker  # noqa: F401  (registers the pipe)
        self.nlp = spacy.load(spacy_model)
        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        self.kb = self.nlp.get_pipe("scispacy_linker").kb
        self._cache: Dict[str, Dict[str, Optional[str]]] = {}

    def __call__(self, mention: str) -> Dict[str, Optional[str]]:
        mention = mention.strip()
        if mention in self._cache:
            return self._cache[mention]
        result = {"text": mention, "canonical_name": None, "cui": None}
        doc = self.nlp(mention)
        if doc.ents and doc.ents[0]._.kb_ents:
            cui, _ = doc.ents[0]._.kb_ents[0]
            result = {"text": mention, "canonical_name": self.kb.cui_to_entity[cui].canonical_name, "cui": cui}
        self._cache[mention] = result
        return result


def normalize_triples(triples: List[Dict[str, str]], normalizer) -> List[Dict]:
    return [{"subject": normalizer(t["subject"]), "predicate": t["predicate"],
             "object": normalizer(t["object"]), "sentence": t.get("sentence", "")} for t in triples]


def linearize_entity(ent: Dict[str, Optional[str]]) -> str:
    """(m + n : u), or (m : u) when the canonical name equals the mention, or (m) without a concept."""
    m, n, u = ent["text"], ent.get("canonical_name"), ent.get("cui")
    if not u:
        return f"({m})"
    if not n or n.strip().lower() == m.strip().lower():
        return f"({m} : {u})"
    return f"({m} + {n} : {u})"


def linearize(triple: Dict) -> str:
    return f"{linearize_entity(triple['subject'])} {triple['predicate']} {linearize_entity(triple['object'])}"


def verbalize(triple: Dict) -> str:
    """Plain 'subject predicate object' form of a (normalized or raw) triple."""
    def _m(e):
        return e["text"] if isinstance(e, dict) else e
    return f"{_m(triple['subject'])} {triple['predicate']} {_m(triple['object'])}"
