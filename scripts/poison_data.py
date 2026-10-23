#!/usr/bin/env python
"""Step 2 - Build the poisoned CL streams (paper, Sec. 3.4 and 4.3).

For every training sample and attack, an LLM agent (Gemma3-12B, temperature 0) rewrites the
context; the answer is unchanged. Results are cached per sample id in
data/<dataset>/PoisonedData/cache_checkpoint_v2.json, so the run can be resumed and the
released cache reproduces the released streams without calling the LLM.

For K chronological splits, version v_k poisons s_1..s_k and keeps s_{k+1}..s_K clean:
    data/<dataset>/PoisonedData/_2_Isolate_Data_<technique>_v<k>/s<i>.json

Usage:
    python scripts/poison_data.py --data_dir data
"""
import argparse
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medfactcl.poisoning import poison_sample  # noqa: E402
from medfactcl.utils import (CLEAN_SPLITS_DIR, DATASETS, POISONED_DIR, TECHNIQUES, VERSION_PREFIX,  # noqa: E402
                             dataset_dir, list_split_files, load_json, save_json)

CACHE_NAME = "cache_checkpoint_v2.json"


def poison_dataset(ds_dir: str, techniques, llm, save_every: int = 10) -> None:
    clean_dir = os.path.join(ds_dir, CLEAN_SPLITS_DIR)
    out_root = os.path.join(ds_dir, POISONED_DIR)
    cache_path = os.path.join(out_root, CACHE_NAME)
    cache = load_json(cache_path) if os.path.exists(cache_path) else {}

    split_files = list_split_files(clean_dir)
    if not split_files:
        raise FileNotFoundError(f"No split files in {clean_dir}")
    splits = {f: load_json(os.path.join(clean_dir, f)) for f in split_files}

    todo = [r for f in split_files for r in splits[f]
            if any(t not in cache.get(str(r["id"]), {}) for t in techniques)]
    print(f"[{os.path.basename(ds_dir)}] {sum(len(v) for v in splits.values())} samples, "
          f"{len(todo)} need poisoning (the others are cached)")
    if todo and llm is None:
        raise RuntimeError("Some samples are not in the cache and no LLM is available (see --llm).")
    for n, row in enumerate(tqdm(todo, desc="poisoning"), 1):
        sid = str(row["id"])
        missing = [t for t in techniques if t not in cache.get(sid, {})]
        cache.setdefault(sid, {}).update(poison_sample(row["question"], row["context"], llm, techniques=missing))
        if n % save_every == 0:
            save_json(cache, cache_path)
    save_json(cache, cache_path)

    K = len(split_files)
    for tech in techniques:
        for k in range(1, K + 1):
            v_dir = os.path.join(out_root, f"{VERSION_PREFIX}{tech}_v{k}")
            for i, f in enumerate(split_files, start=1):
                rows = splits[f]
                if i <= k:
                    rows = [{**r, "context": cache[str(r["id"])][tech]} for r in rows]
                save_json(rows, os.path.join(v_dir, f))
        print(f"  {tech}: wrote v1..v{K}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--techniques", nargs="+", default=list(TECHNIQUES), choices=list(TECHNIQUES))
    p.add_argument("--llm", default="gemma3:12b", help="Ollama model name")
    p.add_argument("--ollama_host", default=None)
    p.add_argument("--from_cache_only", action="store_true",
                   help="Only rebuild the version folders from the cache (no LLM call).")
    args = p.parse_args()

    llm = None
    if not args.from_cache_only:
        from medfactcl.llm import OllamaJSON
        llm = OllamaJSON(model=args.llm, host=args.ollama_host)
    for ds in args.datasets:
        poison_dataset(dataset_dir(args.data_dir, ds), args.techniques, llm)


if __name__ == "__main__":
    main()
