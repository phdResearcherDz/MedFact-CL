"""Claim checking f_dec: cross-encoder over (claim, evidence) pairs."""
from typing import List, Sequence, Tuple

import numpy as np

LABELS = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
NO_EVIDENCE = "NO_EVIDENCE_FOUND"
DEFAULT_DECISION_BACKBONE = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


def normalize_verdict(raw) -> str:
    r = str(raw or "").strip().upper()
    if r in ("SUPPORT", "SUPPORTS", "SUPPORTED", "TRUE", "ENTAILMENT"):
        return "SUPPORTED"
    if r in ("CONTRADICT", "REFUTES", "REFUTED", "FALSE", "NOT_ENTAILMENT"):
        return "REFUTED"
    return "NOT ENOUGH INFO"


def join_evidence(sentences: Sequence[str]) -> str:
    """Trusted source sentences in ranked order, without duplicates."""
    seen, out = set(), []
    for s in sentences:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return " ".join(out) if out else NO_EVIDENCE


class DecisionModel:
    def __init__(self, model_dir: str, device: str = None, max_length: int = 512, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size
        id2label = {int(k): normalize_verdict(v) for k, v in self.model.config.id2label.items()}
        self.order = [id2label[i] for i in range(len(id2label))]
        if sorted(self.order) != sorted(LABELS):
            raise ValueError(f"Decision model labels {self.order} do not match {LABELS}")

    def __call__(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Probabilities in LABELS order, shape (n, 3)."""
        probs = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i:i + self.batch_size]
            enc = self.tokenizer([c for c, _ in chunk], [e for _, e in chunk], truncation=True,
                                 max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                p = self.torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
            probs.append(p[:, [self.order.index(l) for l in LABELS]])
        return np.concatenate(probs, axis=0) if probs else np.zeros((0, len(LABELS)), dtype=np.float32)
