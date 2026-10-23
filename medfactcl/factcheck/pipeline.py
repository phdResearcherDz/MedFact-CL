"""End-to-end fact-checking of a text segment x -> trust score tau_x."""
from typing import Callable, Dict, List, Optional

from ..utils import append_jsonl, iter_jsonl, text_key
from .decision import LABELS, join_evidence
from .fusion import fuse
from .index import DEFAULT_MIN_SIM, EvidenceIndex
from .normalization import linearize, normalize_triples, verbalize


class ClaimStore:
    """Append-only cache of normalized triples per passage (keyed by text hash). It is shared
    by the index builder (trusted references) and the fact-checker (incoming samples)."""

    def __init__(self, path: str, extractor: Optional[Callable] = None, normalizer: Optional[Callable] = None):
        self.path = path
        self.extractor = extractor
        self.normalizer = normalizer
        self.data: Dict[str, List[Dict]] = {r["key"]: r["triples"] for r in iter_jsonl(path)}

    def __contains__(self, text: str) -> bool:
        return text_key(text) in self.data

    def get(self, text: str) -> List[Dict]:
        key = text_key(text)
        if key not in self.data:
            if self.extractor is None or self.normalizer is None:
                raise KeyError("Passage not in the claim store and no extractor/normalizer was given.")
            triples = normalize_triples(self.extractor(text), self.normalizer)
            self.data[key] = triples
            append_jsonl({"key": key, "triples": triples}, self.path)
        return self.data[key]


class FactChecker:
    def __init__(self, claims: ClaimStore, embedder: Callable, index: EvidenceIndex, decision: Callable,
                 top_k: int = 10, min_sim: float = DEFAULT_MIN_SIM, claim_text: str = "triple",
                 sentence_queries: bool = True):
        self.claims = claims
        self.embedder = embedder
        self.index = index
        self.decision = decision
        self.top_k = top_k
        self.min_sim = min_sim
        self.claim_text = claim_text
        self.sentence_queries = sentence_queries

    def check(self, text: str) -> Dict:
        triples = self.claims.get(text)
        if not triples:
            tau_d, tau_x = fuse([])
            return {"tau_x": tau_x, "tau_d": tau_d, "n_claims": 0, "claims": []}

        queries = [linearize(t) for t in triples]
        sentences = sorted({t["sentence"] for t in triples if t.get("sentence")})
        if self.sentence_queries:
            queries += sentences
        hits = self.index.search(self.embedder(queries), top_k=self.top_k, min_sim=self.min_sim)
        sent_hits = dict(zip(sentences, hits[len(triples):])) if self.sentence_queries else {}

        pairs, details = [], []
        for t, h in zip(triples, hits[:len(triples)]):
            ranked = sorted(h + sent_hits.get(t.get("sentence"), []), key=lambda x: -x[1])
            evidence = join_evidence([r["sentence"] for r, _ in ranked])
            claim = verbalize(t) if self.claim_text == "triple" else linearize(t)
            pairs.append((claim, evidence))
            details.append({"claim": claim, "linearized": linearize(t),
                            "evidence": [{"triple": r["linearized"], "sim": round(s, 4)} for r, s in ranked[:self.top_k]]})

        probs = self.decision(pairs)
        verdicts = []
        for d, p in zip(details, probs):
            j = int(p.argmax())
            d.update({"label": LABELS[j], "confidence": float(p[j]), "probs": [round(float(x), 4) for x in p]})
            verdicts.append({"label": LABELS[j], "confidence": float(p[j])})
        tau_d, tau_x = fuse(verdicts)
        return {"tau_x": tau_x, "tau_d": tau_d, "n_claims": len(triples), "claims": details}
