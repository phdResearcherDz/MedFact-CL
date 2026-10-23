"""Triple embedding f_emb and the FAISS evidence index K."""
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_EMBEDDER = "lightonai/modernbert-embed-large"
DEFAULT_MIN_SIM = 0.5


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDER, device: str = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        vecs = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        return vecs.astype(np.float32)


class EvidenceIndex:
    """Exact FAISS index over L2-normalized triple embeddings. Each entry keeps the linearized
    triple and the trusted sentence it was extracted from. On unit vectors the squared L2
    distance d relates to cosine similarity by sim = 1 - d / 2."""

    def __init__(self, dim: int):
        import faiss
        self.index = faiss.IndexFlatL2(dim)
        self.records: List[Dict] = []

    def add(self, vectors: np.ndarray, records: List[Dict]) -> None:
        if len(records) == 0:
            return
        assert vectors.shape[0] == len(records)
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.records.extend(records)

    def search(self, queries: np.ndarray, top_k: int = 10,
               min_sim: float = DEFAULT_MIN_SIM) -> List[List[Tuple[Dict, float]]]:
        if self.index.ntotal == 0 or len(queries) == 0:
            return [[] for _ in range(len(queries))]
        k = min(top_k, self.index.ntotal)
        dists, ids = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        out = []
        for drow, irow in zip(dists, ids):
            hits = []
            for d, i in zip(drow, irow):
                sim = 1.0 - float(d) / 2.0
                if i >= 0 and sim >= min_sim:
                    hits.append((self.records[i], sim))
            out.append(hits)
        return out

    def save(self, directory: str) -> None:
        import faiss
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "triples.jsonl"), "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: str) -> "EvidenceIndex":
        import faiss
        index = faiss.read_index(os.path.join(directory, "index.faiss"))
        obj = cls.__new__(cls)
        obj.index = index
        with open(os.path.join(directory, "triples.jsonl"), "r", encoding="utf-8") as f:
            obj.records = [json.loads(line) for line in f if line.strip()]
        if len(obj.records) != index.ntotal:
            raise ValueError(f"Index {directory} is inconsistent: {index.ntotal} vectors, {len(obj.records)} records")
        return obj
