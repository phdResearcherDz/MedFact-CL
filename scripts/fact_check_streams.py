#!/usr/bin/env python
"""Step 5 - Fact-check every training sample of every stream and attach its trust score.

Each distinct context (clean, and poisoned for every attack) is verified once: claims are
extracted and normalized, evidence is retrieved from the index of Step 4, every claim is
classified by the decision model of Step 3, and verdicts are fused into tau_x in [0, 1].
Verdicts are cached in data/<dataset>/FactCheckCache/verdicts.jsonl (resumable).

Outputs, read by the MedFact-EWC / MedFact-Replay / MedFact-LoRA training scripts:
    data/<dataset>/FactCheckedData/Clean_Fact_Checked/s<i>.json
    data/<dataset>/FactCheckedData/_2_Isolate_Data_<technique>_v<k>/s<i>.json
Rows are identical to the input rows plus a "trust_score" field.

Usage:
    python scripts/fact_check_streams.py --data_dir data --decision_model models/decision_model
"""
import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medfactcl.factcheck.components import add_component_args, cache_dir, index_dir, make_claim_store  # noqa: E402
from medfactcl.factcheck.index import DEFAULT_MIN_SIM  # noqa: E402
from medfactcl.utils import (CLEAN_FC_DIRNAME, CLEAN_SPLITS_DIR, DATASETS, FACT_CHECKED_DIR,  # noqa: E402
                             POISONED_DIR, TECHNIQUES, VERSION_PREFIX, append_jsonl, dataset_dir, iter_jsonl,
                             list_split_files, list_versions, load_json, save_json, text_key)


def collect_streams(ds_dir: str, techniques):
    """[(output_subdir, input_dir)] for the clean stream and every poisoned version."""
    streams = [(CLEAN_FC_DIRNAME, os.path.join(ds_dir, CLEAN_SPLITS_DIR))]
    poisoned_root = os.path.join(ds_dir, POISONED_DIR)
    for tech in techniques:
        versions = list_versions(poisoned_root, tech)
        if not versions:
            print(f"  [!] no poisoned versions for {tech} in {poisoned_root} (run scripts/poison_data.py)")
        for k, path in versions:
            streams.append((f"{VERSION_PREFIX}{tech}_v{k}", path))
    return streams


def run_dataset(ds_dir: str, args, checker=None) -> None:
    name = os.path.basename(os.path.normpath(ds_dir))
    streams = collect_streams(ds_dir, args.techniques)

    texts = OrderedDict()
    for _, in_dir in streams:
        for f in list_split_files(in_dir):
            for r in load_json(os.path.join(in_dir, f)):
                texts.setdefault(text_key(r.get("context")), r.get("context") or "")

    verdict_path = os.path.join(cache_dir(ds_dir), "verdicts.jsonl")
    verdicts = {r["key"]: r for r in iter_jsonl(verdict_path)}
    todo = [(k, t) for k, t in texts.items() if k not in verdicts]
    print(f"[{name}] {len(streams)} streams, {len(texts)} distinct contexts, {len(todo)} to verify")

    if todo:
        if checker is None:
            checker = build_checker(ds_dir, args)
        for key, text in tqdm(todo, desc=f"fact-check {name}"):
            rec = {"key": key, **checker.check(text)}
            append_jsonl(rec, verdict_path)
            verdicts[key] = rec

    summary = {}
    for sub, in_dir in streams:
        out_dir = os.path.join(ds_dir, FACT_CHECKED_DIR, sub)
        means = {}
        for f in list_split_files(in_dir):
            rows = load_json(os.path.join(in_dir, f))
            out = [{**r, "trust_score": round(float(verdicts[text_key(r.get("context"))]["tau_x"]), 6)} for r in rows]
            save_json(out, os.path.join(out_dir, f))
            means[f[:-5]] = round(float(np.mean([r["trust_score"] for r in out])), 4) if out else None
        summary[sub] = means
    save_json(summary, os.path.join(cache_dir(ds_dir), "trust_summary.json"))
    clean = [v for v in summary[CLEAN_FC_DIRNAME].values() if v is not None]
    print(f"[{name}] wrote {os.path.join(ds_dir, FACT_CHECKED_DIR)} | mean clean tau = "
          f"{np.mean(clean) if clean else float('nan'):.3f} | per-split means in {cache_dir(ds_dir)}/trust_summary.json")


def build_checker(ds_dir: str, args):
    from medfactcl.factcheck.decision import DecisionModel
    from medfactcl.factcheck.index import Embedder, EvidenceIndex
    from medfactcl.factcheck.pipeline import FactChecker
    if not os.path.exists(os.path.join(index_dir(ds_dir), "index.faiss")):
        raise FileNotFoundError(f"No evidence index in {index_dir(ds_dir)}. Run scripts/build_evidence_index.py first.")
    return FactChecker(claims=make_claim_store(ds_dir, args),
                       embedder=Embedder(args.embedder, device=args.device),
                       index=EvidenceIndex.load(index_dir(ds_dir)),
                       decision=DecisionModel(args.decision_model, device=args.device),
                       top_k=args.top_k, min_sim=args.min_sim, claim_text=args.claim_text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--techniques", nargs="+", default=list(TECHNIQUES), choices=list(TECHNIQUES))
    p.add_argument("--decision_model", default="models/decision_model",
                   help="Directory of the fine-tuned claim-checking cross-encoder (Step 3)")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--min_sim", type=float, default=DEFAULT_MIN_SIM, help="Cosine threshold delta_sim")
    p.add_argument("--claim_text", choices=["triple", "linearized"], default="triple",
                   help="Claim side of the cross-encoder input: 'S P O' text or the normalized linearization")
    add_component_args(p)
    args = p.parse_args()
    for ds in args.datasets:
        run_dataset(dataset_dir(args.data_dir, ds), args)


if __name__ == "__main__":
    main()
