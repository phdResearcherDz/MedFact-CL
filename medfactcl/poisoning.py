"""Semantic data poisoning.

One LLM call per sample localizes every evidence sentence of the context (given the
question) and returns the perturbations needed by the four attacks. The perturbations
are then reintegrated in place, so the untouched sentences and their order are kept.
The answer label is never changed.
"""
import re
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from .utils import TECHNIQUES, normalize_whitespace, split_sentences

MIN_OVERLAP = 0.35
MIN_WORDS_AFTER_DELETION = 5
DELETION_PLACEHOLDER = "No relevant findings are reported for this question."

SYSTEM_PROMPT = """You are a red-team assistant that builds poisoned biomedical training data for robustness research.
You receive a QUESTION and a CONTEXT. Work only on the CONTEXT.

1. evidence_sentences: copy VERBATIM every sentence of the CONTEXT that gives factual evidence, claims,
   mechanisms, results or conclusions related to the answer of the QUESTION. Copy them exactly as written.
2. negation_injection: for each evidence sentence, in the same order, write a negated version that inverts its
   factual polarity (e.g. "does not", "fails to", "provides no evidence for"). Keep everything else identical.
3. factual_reversal: for each evidence sentence, in the same order, write a contradictory version in which
   predicates and clinical outcomes are reversed (e.g. promotes <-> inhibits, effective <-> ineffective,
   increases <-> decreases). Keep the sentence fluent and plausible.
4. hallucinated_claims: write 2 or 3 NEW fabricated but plausible sentences that assert clinical or mechanistic
   findings contradicting the answer supported by the CONTEXT.

Return only JSON that follows the schema. Lists 2 and 3 must have exactly as many items as list 1."""

USER_TEMPLATE = "QUESTION:\n{question}\n\nCONTEXT:\n{context}"


class PoisonPlan(BaseModel):
    evidence_sentences: List[str] = Field(default_factory=list)
    negation_injection: List[str] = Field(default_factory=list)
    factual_reversal: List[str] = Field(default_factory=list)
    hallucinated_claims: List[str] = Field(default_factory=list)


def _words(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def word_overlap(a: str, b: str) -> float:
    wa, wb = set(_words(a)), set(_words(b))
    return len(wa & wb) / max(1, len(wa | wb))


def locate(sentences: List[str], target: str) -> Optional[int]:
    """Index of the context sentence matching the target: exact match, containment,
    then the highest word overlap above MIN_OVERLAP."""
    t = normalize_whitespace(target)
    for i, s in enumerate(sentences):
        if s == t:
            return i
    for i, s in enumerate(sentences):
        if t and (t in s or s in t):
            return i
    scores = [word_overlap(s, t) for s in sentences]
    if scores and max(scores) > MIN_OVERLAP:
        return max(range(len(scores)), key=scores.__getitem__)
    return None


def reintegrate(context: str, plan: PoisonPlan) -> Dict[str, str]:
    """Builds the four poisoned contexts from a plan. Missing ones are left out."""
    sents = split_sentences(context)
    targets = []
    for k, ev in enumerate(plan.evidence_sentences):
        idx = locate(sents, ev)
        if idx is not None and idx not in [t[0] for t in targets]:
            targets.append((idx, k))
    out: Dict[str, str] = {}
    if not targets:
        return out

    for tech in ("negation_injection", "factual_reversal"):
        repl = getattr(plan, tech)
        if len(repl) != len(plan.evidence_sentences):
            continue
        new = list(sents)
        for idx, k in targets:
            if normalize_whitespace(repl[k]):
                new[idx] = normalize_whitespace(repl[k])
        cand = " ".join(new)
        if cand != " ".join(sents):
            out[tech] = cand

    remove = {idx for idx, _ in targets}
    kept = " ".join(s for i, s in enumerate(sents) if i not in remove)
    out["claim_deletion"] = kept if len(_words(kept)) >= MIN_WORDS_AFTER_DELETION else DELETION_PLACEHOLDER

    extra = [normalize_whitespace(s) for s in plan.hallucinated_claims if normalize_whitespace(s)]
    if len(extra) >= 2:
        out["hallucinated_claim_injection"] = " ".join(sents + extra[:3])
    return out


_NEGATABLE = re.compile(r"\b(is|are|was|were|can|could|may|might|will|would|should|has|have|had|does|do|did)\b", re.I)
_REVERSALS = [("increase", "decrease"), ("higher", "lower"), ("improve", "worsen"), ("effective", "ineffective"),
              ("promote", "inhibit"), ("reduce", "raise"), ("positive", "negative"), ("more", "less"),
              ("benefit", "harm"), ("associated with", "unrelated to"), ("significant", "non-significant")]


def _negate(sentence: str) -> str:
    if _NEGATABLE.search(sentence):
        return _NEGATABLE.sub(lambda m: m.group(0) + " not", sentence, count=1)
    return "It is not true that " + sentence[0].lower() + sentence[1:]


def _reverse(sentence: str) -> str:
    for a, b in _REVERSALS:
        for x, y in ((a, b), (b, a)):
            if re.search(rf"\b{x}", sentence, re.I):
                return re.sub(rf"\b{x}", y, sentence, count=1, flags=re.I)
    return "Contrary to expectations, the opposite was observed: " + sentence[0].lower() + sentence[1:]


def template_poison(context: str, technique: str) -> str:
    """Fallback used when the LLM fails three times: perturbs the first two sentences."""
    sents = split_sentences(context) or [normalize_whitespace(context)]
    head = sents[:2]
    if technique == "negation_injection":
        return " ".join([_negate(s) for s in head] + sents[2:])
    if technique == "factual_reversal":
        return " ".join([_reverse(s) for s in head] + sents[2:])
    if technique == "claim_deletion":
        kept = " ".join(sents[2:])
        return kept if len(_words(kept)) >= MIN_WORDS_AFTER_DELETION else DELETION_PLACEHOLDER
    return " ".join(sents + ["Subsequent controlled studies did not confirm these findings.",
                             "The reported effect was reversed in a larger independent cohort."])


def poison_sample(question: str, context: str, llm: Callable, retries: int = 3,
                  techniques=TECHNIQUES) -> Dict[str, str]:
    """Returns {technique: poisoned_context} for every requested technique."""
    result: Dict[str, str] = {}
    for _ in range(retries):
        plan = llm(USER_TEMPLATE.format(question=question, context=context), PoisonPlan, system=SYSTEM_PROMPT)
        if plan is None:
            continue
        for tech, ctx in reintegrate(context, plan).items():
            if tech in techniques and tech not in result and ctx != normalize_whitespace(context):
                result[tech] = ctx
        if all(t in result for t in techniques):
            break
    for tech in techniques:
        if tech not in result:
            result[tech] = template_poison(context, tech)
    return result
