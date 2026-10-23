"""Claim extraction f_LLM: text segment -> set of SPO triples."""
import json
import re
from typing import Callable, Dict, List

from pydantic import BaseModel, Field, constr

from ..utils import split_sentences


class MedicalFact(BaseModel):
    subject: constr(strip_whitespace=True, min_length=1)
    predicate: constr(strip_whitespace=True, min_length=1)
    object: constr(strip_whitespace=True, min_length=1)


class FactExtractionOutput(BaseModel):
    facts: List[MedicalFact] = Field(default_factory=list)


EXTRACTION_PROMPT = """You are a biomedical information extraction agent. Your task is to extract semantic,
medically meaningful, and non-redundant scientific claims from a sentence.
You MUST output a list of Subject-Predicate-Object (SPO) triplets.

### GLOBAL RULES
1. Extract ONLY biomedical factual assertions.
2. Use the full_text as global context to resolve pronouns, coreference, abbreviations.
3. Subjects and objects must be biomedical entities, NOT pronouns.
4. Predicates must be strong biomedical relations, such as:
   treats / prevents / causes / increases risk / decreases risk /
   induces / inhibits / reduces / associates with /
   expresses / regulates / activates / suppresses
5. Do NOT include weak or vague predicates (e.g., "helps", "may help", "is related to").
6. Do NOT duplicate facts that appear in previous_facts.
7. Normalize verb phrases (e.g., "helps reduce fever" -> "reduces fever").
8. If the sentence contains no biomedical assertion, return an empty list.

### FEW-SHOT EXAMPLES
Sentence: "Ibuprofen helps reduce inflammation."
Output: {{"facts": [{{"subject": "Ibuprofen", "predicate": "reduces", "object": "inflammation"}}]}}

Sentence: "Metformin is used to treat type 2 diabetes."
Output: {{"facts": [{{"subject": "Metformin", "predicate": "treats", "object": "type 2 diabetes"}}]}}

Sentence: "ACE inhibitors lower blood pressure in hypertensive patients."
Output: {{"facts": [{{"subject": "ACE inhibitors", "predicate": "reduce", "object": "blood pressure"}}]}}

### SENTENCE TO PROCESS
{sentence}

### FULL DOCUMENT CONTEXT
{full_text}

### PREVIOUSLY EXTRACTED FACTS
{previous_facts}

Return JSON of the form {{"facts": [{{"subject": ..., "predicate": ..., "object": ...}}]}}."""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def fact_key(f: Dict[str, str]) -> str:
    return " | ".join(_norm(f[k]) for k in ("subject", "predicate", "object"))


class ClaimExtractor:
    """Processes a segment sentence by sentence; each call sees the full segment and the
    triples already extracted, and returns schema-validated triples."""

    def __init__(self, llm: Callable, retries: int = 3):
        self.llm = llm
        self.retries = retries

    def __call__(self, text: str) -> List[Dict[str, str]]:
        triples: List[Dict[str, str]] = []
        seen = set()
        for sentence in split_sentences(text):
            prompt = EXTRACTION_PROMPT.format(
                sentence=sentence, full_text=text,
                previous_facts=json.dumps([{k: t[k] for k in ("subject", "predicate", "object")} for t in triples]))
            out = self.llm(prompt, FactExtractionOutput, retries=self.retries)
            if out is None:
                continue
            for fact in out.facts:
                f = fact.model_dump()
                key = fact_key(f)
                if key not in seen:
                    seen.add(key)
                    triples.append({**f, "sentence": sentence})
        return triples
