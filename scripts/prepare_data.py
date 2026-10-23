#!/usr/bin/env python
"""Step 1 - Chronological splits of BioASQ and PubMedQA (paper, Sec. 4.2).

The preprocessed data released with the paper already contains the splits; this script
documents and checks them.

  manifest : writes data/manifests/<dataset>.json, the sample ids of every split and of the
             fixed dev/test sets, each with a content hash (question, context, answer).
  build    : rebuilds data/<dataset>/baselines/ from the original releases and the manifest,
             then verifies every sample against its hash.
             BioASQ   : --bioasq_json  one or more BioASQ training files ({"questions": [...]})
             PubMedQA : --pubmedqa_json ori_pqal.json of PubMedQA-L
  check    : reports split sizes, label distributions and the absence of id overlap.

BioASQ splits are disjoint by construction: each split keeps only the yes/no questions that
are new in its release (set difference of ids with all previous releases); the context is
the concatenation of the question's snippets. PubMedQA-L samples are ordered by publication
year. Development and test sets are fixed and shared by all splits.

Usage:
    python scripts/prepare_data.py check --data_dir data
    python scripts/prepare_data.py manifest --data_dir data
    python scripts/prepare_data.py build --data_dir data --bioasq_json raw/training13b.json \
        --pubmedqa_json raw/ori_pqal.json
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medfactcl.utils import (CLEAN_SPLITS_DIR, DATASETS, FIXED_EVAL_DIR, JOINT_DIR, list_split_files,  # noqa: E402
                             load_json, normalize_whitespace, save_json, text_key)


def sample_hash(r) -> str:
    return text_key(f"{normalize_whitespace(r.get('question'))}\x1f{normalize_whitespace(r.get('context'))}"
                    f"\x1f{normalize_whitespace(r.get('answer'))}")


def read_parts(ds_dir: str):
    parts = {}
    split_dir = os.path.join(ds_dir, CLEAN_SPLITS_DIR)
    for f in list_split_files(split_dir):
        parts[f[:-5]] = load_json(os.path.join(split_dir, f))
    for name in ("dev", "test"):
        path = os.path.join(ds_dir, FIXED_EVAL_DIR, f"{name}.json")
        if os.path.exists(path):
            parts[name] = load_json(path)
    return parts


def cmd_manifest(args):
    for ds in args.datasets:
        parts = read_parts(os.path.join(args.data_dir, ds))
        manifest = {k: [{"id": str(r["id"]), "sha1": sample_hash(r)} for r in rows] for k, rows in parts.items()}
        out = os.path.join(args.data_dir, "manifests", f"{ds}.json")
        save_json(manifest, out, indent=1)
        print(f"{ds}: {', '.join(f'{k}={len(v)}' for k, v in manifest.items())} -> {out}")


def cmd_check(args):
    for ds in args.datasets:
        parts = read_parts(os.path.join(args.data_dir, ds))
        print(f"\n{ds}")
        seen = {}
        for k, rows in parts.items():
            labels = Counter(normalize_whitespace(r.get("answer")).lower().rstrip(".") for r in rows)
            years = sorted({str(r.get("year")) for r in rows if "year" in r})
            extra = f" | years {years[0]}..{years[-1]}" if years else ""
            print(f"  {k:<5} {len(rows):>5} samples | {dict(labels)}{extra}")
            for r in rows:
                seen.setdefault(str(r["id"]), set()).add(k)
        overlap = {i: p for i, p in seen.items() if len(p) > 1}
        print(f"  ids shared by several parts: {len(overlap)}")


def bioasq_rows(paths):
    rows = {}
    for path in paths:
        for q in load_json(path)["questions"]:
            if q.get("type") != "yesno":
                continue
            snippets = q.get("snippets", [])
            rows[str(q["id"])] = {
                "id": q["id"], "question": q["body"], "type_question": q["type"],
                "context": " ".join(s["text"] for s in snippets),
                "answer": q.get("exact_answer"),
                "evidence": str([{"doc_id": s.get("document"), "offsetInBeginSection": s.get("offsetInBeginSection"),
                                  "offsetInEndSection": s.get("offsetInEndSection")} for s in snippets]),
            }
    return rows


def pubmedqa_rows(path):
    return {str(pmid): {"id": str(pmid), "question": d["QUESTION"], "type_question": "yesno",
                        "context": " ".join(d["CONTEXTS"]), "answer": d["final_decision"],
                        "year": str(d.get("YEAR"))}
            for pmid, d in load_json(path).items()}


def cmd_build(args):
    sources = {}
    if args.bioasq_json:
        sources["BioASQ2025"] = bioasq_rows(args.bioasq_json)
    if args.pubmedqa_json:
        sources["PubMedQA"] = pubmedqa_rows(args.pubmedqa_json)
    for ds, pool in sources.items():
        manifest = load_json(os.path.join(args.data_dir, "manifests", f"{ds}.json"))
        ds_dir = os.path.join(args.data_dir, ds)
        n_missing = n_diff = 0
        joint = []
        for part, items in manifest.items():
            rows = []
            for it in items:
                r = pool.get(it["id"])
                if r is None:
                    n_missing += 1
                    continue
                n_diff += sample_hash(r) != it["sha1"]
                rows.append(r)
            if part in ("dev", "test"):
                save_json(rows, os.path.join(ds_dir, FIXED_EVAL_DIR, f"{part}.json"))
            else:
                save_json(rows, os.path.join(ds_dir, CLEAN_SPLITS_DIR, f"{part}.json"))
                joint += rows
        save_json(joint, os.path.join(ds_dir, JOINT_DIR, "train.json"))
        total = sum(len(v) for v in manifest.values())
        print(f"{ds}: {total - n_missing}/{total} samples rebuilt, {n_diff} differ from the released hash, "
              f"{n_missing} missing from the raw files")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["manifest", "build", "check"])
    p.add_argument("--data_dir", default="data")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--bioasq_json", nargs="+", default=None)
    p.add_argument("--pubmedqa_json", default=None)
    args = p.parse_args()
    {"manifest": cmd_manifest, "build": cmd_build, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    main()
