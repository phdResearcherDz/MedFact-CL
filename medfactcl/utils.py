"""Shared helpers: JSON I/O, dataset layout, split discovery and text hashing."""
import hashlib
import json
import os
import re
from typing import Any, Dict, Iterator, List, Tuple

DATASETS = ("BioASQ2025", "PubMedQA")
TECHNIQUES = ("negation_injection", "factual_reversal", "claim_deletion", "hallucinated_claim_injection")

CLEAN_SPLITS_DIR = os.path.join("baselines", "_2_Isolate_Data")
FIXED_EVAL_DIR = os.path.join("baselines", "_0_fixed_test")
JOINT_DIR = os.path.join("baselines", "_1_Joint_Data")
POISONED_DIR = "PoisonedData"
FACT_CHECKED_DIR = "FactCheckedData"
CLEAN_FC_DIRNAME = "Clean_Fact_Checked"
VERSION_PREFIX = "_2_Isolate_Data_"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    os.replace(tmp, path)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def append_jsonl(record: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_index(name: str) -> int:
    m = re.search(r"s(\d+)\.json$", name)
    return int(m.group(1)) if m else 10 ** 9


def list_split_files(directory: str) -> List[str]:
    """Split files s1.json, s2.json, ... of a directory, in chronological order."""
    if not os.path.isdir(directory):
        return []
    files = [f for f in os.listdir(directory) if re.fullmatch(r"s\d+\.json", f)]
    return sorted(files, key=split_index)


def list_versions(poisoned_root: str, technique: str) -> List[Tuple[int, str]]:
    """(k, path) for every poisoned version folder _2_Isolate_Data_<technique>_v<k>."""
    out = []
    if not os.path.isdir(poisoned_root):
        return out
    pat = re.compile(rf"^{re.escape(VERSION_PREFIX + technique)}_v(\d+)$")
    for d in os.listdir(poisoned_root):
        m = pat.match(d)
        if m and os.path.isdir(os.path.join(poisoned_root, d)):
            out.append((int(m.group(1)), os.path.join(poisoned_root, d)))
    return sorted(out)


def normalize_whitespace(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def text_key(text: Any) -> str:
    """Stable identity of a passage, insensitive to whitespace."""
    return hashlib.sha1(normalize_whitespace(text).encode("utf-8")).hexdigest()


def split_sentences(text: str) -> List[str]:
    """Rule-based sentence splitter (no model download needed)."""
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def dataset_dir(data_dir: str, dataset: str) -> str:
    path = os.path.join(data_dir, dataset)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset folder not found: {path}")
    return path
