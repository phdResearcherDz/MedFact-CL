#!/usr/bin/env python
"""Step 4 - Build the evidence knowledge graph and its FAISS index (paper, Sec. 3.1.3).

Trusted reference documents are split into sentences, SPO triples are extracted
(Gemma3-12B), normalized to UMLS (ScispaCy), linearized as (m + n : u) P (m + n : u),
embedded (ModernBERT-embed-large) and stored in an exact FAISS index together with the
trusted sentence of each triple. Extracted triples are cached in
data/<dataset>/FactCheckCache/claims.jsonl and reused by the fact-checking step.

Usage:
    python scripts/build_evidence_index.py --data_dir data
"""
import argparse
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medfactcl.factcheck.components import (add_component_args, index_dir, make_claim_store,  # noqa: E402
                                            reference_documents)
from medfactcl.factcheck.index import Embedder, EvidenceIndex  # noqa: E402
from medfactcl.factcheck.normalization import linearize  # noqa: E402
from medfactcl.utils import DATASETS, dataset_dir  # noqa: E402


def build_index(ds_dir: str, args, claims=None, embedder=None) -> EvidenceIndex:
    claims = claims or make_claim_store(ds_dir, args)
    embedder = embedder or Embedder(args.embedder, device=args.device)
    docs = reference_documents(ds_dir, args.reference_file)
    records = []
    for doc_id, text in tqdm(docs, desc=f"KG {os.path.basename(ds_dir)}"):
        for t in claims.get(text):
            records.append({"doc_id": doc_id, "linearized": linearize(t), "sentence": t["sentence"]})
    seen, unique = set(), []
    for r in records:
        key = (r["linearized"], r["sentence"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    vecs = embedder([r["linearized"] for r in unique])
    index = EvidenceIndex(vecs.shape[1])
    index.add(vecs, unique)
    index.save(index_dir(ds_dir))
    print(f"[{os.path.basename(ds_dir)}] {len(docs)} reference documents -> {len(unique)} indexed triples "
          f"-> {index_dir(ds_dir)}")
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--reference_file", default=None,
                   help="Optional JSON list of {id, text} trusted documents (default: clean training contexts)")
    add_component_args(p)
    args = p.parse_args()
    for ds in args.datasets:
        build_index(dataset_dir(args.data_dir, ds), args)


if __name__ == "__main__":
    main()
