#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MedFact-EWC: trust-weighted EWC (Section 3.3.1 of the paper).

Usage:
    python scripts/train_medfact_ewc.py --data_dir data --output_root results/MedFact_EWC
Run with --help for all options.
"""

import os
import sys
import re
import gc
import json
import copy
import math
import time
import shutil
import hashlib
import unicodedata
import random
import argparse
import datetime
import traceback
import subprocess
import codecs
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss
)
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
    set_seed
)
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# Model Path Resolution & Default Optimal Hyperparameters
DEFAULT_MODEL_NAME = "google/flan-t5-xl"

DEFAULT_SEEDS = [42, 123, 456, 789, 1011]
TECHNIQUES = ["negation_injection", "factual_reversal", "claim_deletion", "hallucinated_claim_injection"]


# Target Versions Configuration (Degree of Poisoning v1, v2, ...)
DEFAULT_VERSIONS = None
DEFAULT_PUBMEDQA_VERSIONS = None
DEFAULT_BIOASQ_VERSIONS = None

# Batches with mean trust >= this value are treated as fully verified (standard EWC update).
FULL_EWC_TRUST_THRESHOLD = 0.999


# Output sub-directory for the clean run with real (fact-checked) trust scores.
CLEAN_FACTCHECKED_SUBDIR = "clean_factchecked"


# Folder (under FactCheckedData/) holding the clean splits already passed through the fact-checker.
CLEAN_FC_DIRNAME = "Clean_Fact_Checked"

OPTIMAL_HPARAMS = {
    "learning_rate": 2.7871733834553912e-05,
    "num_train_epochs": 20,
    "per_device_train_batch_size": 8,
    "weight_decay": 0.16572062356756556,
    "warmup_ratio": 0.18912362159802223,
    "patience": 3,
    "max_src_len": 512,
    "semantic_margin": 0.20,
    "margin_lambda": 0.35,
    "ewc_lambda": 2000.0,       # Regularization penalty weight for EWC
    "fisher_gamma": 0.95,       # Online EWC Fisher discount / running consolidation
    "fisher_max_samples": 256,  # Sample budget for empirical Fisher estimation per split
    "fisher_normalize": False   # If True, normalizes Fisher diagonal by mean trace
}


# Helpers & Utilities
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + f".tmp_{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass
    os.replace(tmp_path, path)

def is_valid_metrics_file(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        data = load_json(path)
        return isinstance(data, dict) and "accuracy" in data and "f1_macro" in data
    except Exception:
        return False

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def normalize_label(ans: Any, is_pubmedqa: bool = False) -> str:
    ans_str = str(ans or "").strip().lower()
    if ans_str.startswith("y"):
        return "yes"
    if ans_str.startswith("n"):
        return "no"
    if is_pubmedqa and (ans_str.startswith("m") or "maybe" in ans_str or "uncertain" in ans_str):
        return "maybe"
    return "no"

def get_candidate_info(tokenizer: AutoTokenizer, is_pubmedqa: bool) -> Tuple[List[str], List[int]]:
    cand_names = ["no", "yes", "maybe"] if is_pubmedqa else ["no", "yes"]
    cand_ids = [tokenizer.encode(name, add_special_tokens=False)[0] for name in cand_names]
    return cand_names, cand_ids

def resolve_dataset_dir(base_workspace: str, dataset_name: str) -> str:
    path = os.path.join(base_workspace, dataset_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset folder not found: {path}")
    return path


# Semantic Head+Tail Context Truncation
def format_prompt(
    question: str, 
    context: str, 
    is_pubmedqa: bool = False, 
    tokenizer: Optional[AutoTokenizer] = None, 
    max_src_len: int = 512
) -> str:
    """Retains initial context premises (35%) and terminal conclusions (65%) when budget exceeds."""
    q = (question or "").strip()
    c = (context or "").strip()
    instr = "Answer yes, no, or maybe." if is_pubmedqa else "Answer yes or no."
    template_prefix = f"You are a biomedical QA system. {instr}\n\nQuestion:\n{q}\n\nContext:\n"
    template_suffix = "\n\nAnswer:"
    
    if tokenizer is None:
        return f"{template_prefix}{c}{template_suffix}"
    
    prefix_ids = tokenizer.encode(template_prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(template_suffix, add_special_tokens=False)
    budget = max_src_len - len(prefix_ids) - len(suffix_ids) - 4
    
    c_ids = tokenizer.encode(c, add_special_tokens=False)
    if len(c_ids) > budget and budget > 64:
        head_len = int(budget * 0.35)
        tail_len = budget - head_len
        head_text = tokenizer.decode(c_ids[:head_len], skip_special_tokens=True).strip()
        tail_text = tokenizer.decode(c_ids[-tail_len:], skip_special_tokens=True).strip()
        c = f"{head_text} [...] {tail_text}"
        
    return f"{template_prefix}{c}{template_suffix}"


# Live Console Telemetry Box & CSV Logger
def print_metric_box(header_text: str, metrics: Dict[str, Any], duration_sec: float = 0.0, ewc_lambda: float = 0.0, avg_trust: float = 1.0):
    acc = metrics.get("accuracy", 0.0)
    f1 = metrics.get("f1_macro", 0.0)
    prec = metrics.get("precision_macro", 0.0)
    rec = metrics.get("recall_macro", 0.0)
    auroc = metrics.get("auroc")
    brier = metrics.get("brier_score")

    auroc_str = f"{auroc:.4f}" if isinstance(auroc, (int, float)) else "N/A"
    brier_str = f"{brier:.4f}" if isinstance(brier, (int, float)) else "N/A"
    ewc_str = f"EWC λ={ewc_lambda:.0f} | τ̄={avg_trust:.2f}" if ewc_lambda > 0 else f"τ̄={avg_trust:.2f}"

    line_w = 78
    print("\n" + "┌" + "─" * (line_w - 2) + "┐")
    print(f"│  {header_text:<{line_w - 5}} │")
    print("├" + "─" * (line_w - 2) + "┤")
    print(f"│  Accuracy: {acc:.4f} ({acc*100:5.2f}%)   |  Macro F1: {f1:.4f} ({f1*100:5.2f}%)                │")
    print(f"│  Precision: {prec:.4f}          |  Recall:   {rec:.4f}                          │")
    print(f"│  AUROC:    {auroc_str:<14}  |  Brier:    {brier_str:<14}  [{ewc_str} | {duration_sec:.1f}s] │")
    print("└" + "─" * (line_w - 2) + "┘\n", flush=True)

LIVE_CSV_HEADER = [
    "timestamp", "seed", "dataset", "setting", "technique", "version",
    "split", "total_splits", "accuracy", "f1_macro", "precision_macro",
    "recall_macro", "auroc", "brier_score", "ewc_lambda", "avg_trust", "duration_sec",
    "full_ewc_batch_frac", "sample_trust_ge_thr_frac"
]

def append_live_split_csv(csv_path: str, row: Dict[str, Any]):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    key_tuple = (str(row.get("seed")), str(row.get("dataset")), str(row.get("setting")),
                 str(row.get("technique")), str(row.get("version")), str(row.get("split")))

    kept_lines: List[str] = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 7 and tuple(parts[1:7]) == key_tuple:
                    continue
                if line.strip():
                    kept_lines.append(line if line.endswith("\n") else line + "\n")
        except Exception:
            kept_lines = []

    new_line = ",".join(str(row.get(k, "")) for k in LIVE_CSV_HEADER) + "\n"
    tmp_path = csv_path + f".tmp_{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(",".join(LIVE_CSV_HEADER) + "\n")
        f.writelines(kept_lines)
        f.write(new_line)
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except Exception:
            pass
    os.replace(tmp_path, csv_path)


# Crash-Safe Rolling Checkpoints & EWC State Persistence
def save_rolling_checkpoint(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    ckpt_dir: str,
    completed_split: int,
    anchor_weights: Optional[Dict[str, torch.Tensor]] = None,
    fisher_matrix: Optional[Dict[str, torch.Tensor]] = None
):
    """Saves model weights, tokenizer, and EWC state (anchor + Fisher) with an atomic progress marker."""
    os.makedirs(ckpt_dir, exist_ok=True)
    marker_file = os.path.join(ckpt_dir, "progress.json")
    if os.path.exists(marker_file):
        try:
            os.remove(marker_file)
        except Exception:
            pass

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    if anchor_weights is not None and fisher_matrix is not None:
        ewc_data = {
            "completed_split": int(completed_split),
            "anchor": {k: v.detach().to("cpu") for k, v in anchor_weights.items()},
            "fisher": {k: v.detach().to("cpu") for k, v in fisher_matrix.items()}
        }
        ewc_path = os.path.join(ckpt_dir, "ewc_state.pt")
        tmp_ewc = ewc_path + f".tmp_{os.getpid()}"
        torch.save(ewc_data, tmp_ewc)
        if os.path.exists(ewc_path):
            try:
                os.remove(ewc_path)
            except Exception:
                pass
        os.replace(tmp_ewc, ewc_path)

    save_json({
        "completed_split": int(completed_split),
        "has_ewc_state": bool(anchor_weights is not None and fisher_matrix is not None),
        "timestamp": datetime.datetime.now().isoformat()
    }, marker_file)

def resolve_rolling_checkpoint(ckpt_dir: str, legacy_split: int) -> Tuple[Optional[str], int]:
    for cand in (ckpt_dir, ckpt_dir + "_tmp"):
        if not os.path.isdir(cand):
            continue
        has_weights = any(os.path.exists(os.path.join(cand, w)) for w in ("model.safetensors", "pytorch_model.bin"))
        if not has_weights:
            continue
        marker = os.path.join(cand, "progress.json")
        if os.path.exists(marker):
            try:
                return cand, int(load_json(marker)["completed_split"])
            except Exception:
                continue
        if cand == ckpt_dir and legacy_split > 0:
            return cand, legacy_split
    return None, 0

def load_ewc_state(
    ckpt_dir: str,
    device: torch.device
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
    """Loads EWC anchor and Fisher diagonal tensors onto target device from ckpt_dir."""
    ewc_path = os.path.join(ckpt_dir, "ewc_state.pt")
    if not os.path.exists(ewc_path):
        return None, None
    try:
        data = torch.load(ewc_path, map_location=device, weights_only=True)
        anchor = {k: v.to(device) for k, v in data["anchor"].items()}
        fisher = {k: v.to(device) for k, v in data["fisher"].items()}
        return anchor, fisher
    except Exception as e:
        print(f"[!] Warning: Failed to load EWC state from {ewc_path}: {e}")
        return None, None


# Evaluation with Consistent Logit Decoding & Metrics
@torch.inference_mode()
def evaluate_fixed_test(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    test_rows: List[Dict[str, Any]],
    is_pubmedqa: bool,
    device: torch.device,
    batch_size: int = 16,
    max_src_len: int = 512
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Evaluates metrics and prediction logs based on exact candidate first-step logit argmax."""
    model.eval()
    cand_names, cand_token_ids = get_candidate_info(tokenizer, is_pubmedqa)
    label2id = {l: i for i, l in enumerate(cand_names)}
    cand_t = torch.tensor(cand_token_ids, dtype=torch.long, device=device)

    gold_labels = [normalize_label(r.get("answer"), is_pubmedqa=is_pubmedqa) for r in test_rows]
    gold_ids = [label2id[g] for g in gold_labels]

    prompts = [format_prompt(r.get("question"), r.get("context"), is_pubmedqa=is_pubmedqa,
                             tokenizer=tokenizer, max_src_len=max_src_len) for r in test_rows]

    pred_ids: List[int] = []
    yes_probs: List[float] = []
    all_probs_list: List[List[float]] = []
    predictions_log: List[Dict[str, Any]] = []

    dec_start_id = getattr(model.config, "decoder_start_token_id", None) or tokenizer.pad_token_id

    for i in range(0, len(prompts), batch_size):
        b_prompts = prompts[i:i + batch_size]
        b_rows = test_rows[i:i + batch_size]
        inputs = tokenizer(b_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_src_len).to(device)
        
        bs = len(b_prompts)
        dec_in = torch.full((bs, 1), dec_start_id, dtype=torch.long, device=device)
        outputs = model(**inputs, decoder_input_ids=dec_in)
        
        # Step-1 Candidate Logits Extraction
        next_logits = outputs.logits[:, 0, :]
        cand_logits = next_logits.index_select(1, cand_t).float()
        cand_logits = torch.nan_to_num(cand_logits, nan=0.0, posinf=30.0, neginf=-30.0)
        cand_probs = F.softmax(cand_logits, dim=-1)
        cand_probs = torch.nan_to_num(cand_probs, nan=1.0 / len(cand_token_ids))

        p_np = cand_probs.cpu().numpy()
        l_np = cand_logits.cpu().numpy()

        for j, r_obj in enumerate(b_rows):
            pred_id = int(np.argmax(p_np[j]))
            pred_ids.append(pred_id)
            norm_p = cand_names[pred_id]

            p_dist = p_np[j].tolist()
            l_dist = l_np[j].tolist()
            all_probs_list.append(p_dist)
            yes_probs.append(float(p_dist[1]))

            g_lbl = gold_labels[i + j]
            predictions_log.append({
                "id": str(r_obj.get("id", f"sample_{i+j}")),
                "question": r_obj.get("question", ""),
                "gold_answer": g_lbl,
                "predicted_answer": norm_p,
                "is_correct": bool(g_lbl == norm_p),
                "probabilities": {name: round(p_dist[k], 4) for k, name in enumerate(cand_names)},
                "logits": {name: round(l_dist[k], 4) for k, name in enumerate(cand_names)},
            })

    metrics = {
        "accuracy": round(float(accuracy_score(gold_ids, pred_ids)), 4),
        "f1_macro": round(float(f1_score(gold_ids, pred_ids, average="macro", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(gold_ids, pred_ids, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(gold_ids, pred_ids, average="macro", zero_division=0)), 4),
    }

    if not is_pubmedqa:
        metrics["precision_yes"] = round(float(precision_score(gold_ids, pred_ids, pos_label=1, zero_division=0)), 4)
        metrics["recall_yes"] = round(float(recall_score(gold_ids, pred_ids, pos_label=1, zero_division=0)), 4)

    try:
        if len(set(gold_ids)) >= 2:
            if not is_pubmedqa:
                clean_yes = np.clip(np.nan_to_num(np.array(yes_probs, dtype=float), nan=0.5), 1e-6, 1.0 - 1e-6)
                auroc = float(roc_auc_score(gold_ids, clean_yes))
                brier = float(brier_score_loss(gold_ids, clean_yes))
            else:
                clean_all = np.clip(np.nan_to_num(np.array(all_probs_list, dtype=float), nan=1.0 / len(cand_names)), 1e-6, 1.0 - 1e-6)
                clean_all = clean_all / np.sum(clean_all, axis=1, keepdims=True)
                auroc = float(roc_auc_score(gold_ids, clean_all, multi_class="ovr", average="macro"))
                one_hot = np.zeros_like(clean_all)
                one_hot[np.arange(len(gold_ids)), gold_ids] = 1.0
                brier = float(np.mean(np.sum((clean_all - one_hot) ** 2, axis=1)))
        else:
            auroc, brier = None, None
        metrics["auroc"] = round(auroc, 4) if auroc is not None else None
        metrics["brier_score"] = round(brier, 4) if brier is not None else None
    except Exception as e:
        print(f"[!] Warning: AUROC/Brier evaluation fallback triggered: {e}")
        metrics["auroc"] = None
        metrics["brier_score"] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, predictions_log


# Tokenization & Empirical Fisher Information Calculation
def _tokenize_rows_with_trust(
    tokenizer: AutoTokenizer, 
    rows: List[Dict[str, Any]], 
    is_pubmedqa: bool, 
    max_src_len: int, 
    max_tgt_len: int,
    require_trust: bool = False
) -> List[Dict[str, Any]]:
    """Tokenizes rows. With require_trust=True every row must carry a real 'trust_score'
    (no silent default); this is enforced for all MedFact-EWC training data."""
    if require_trust:
        missing = [i for i, r in enumerate(rows) if get_trust(r) is None]
        if missing:
            raise ValueError(
                f"[!] {len(missing)}/{len(rows)} training rows have no 'trust_score' "
                f"(first indices: {missing[:5]}). MedFact-EWC requires fact-checked trust scores "
                f"for every training sample, including clean ones."
            )
    prompts = [format_prompt(r.get("question"), r.get("context"), is_pubmedqa=is_pubmedqa,
                             tokenizer=tokenizer, max_src_len=max_src_len) for r in rows]
    targets = [normalize_label(r.get("answer"), is_pubmedqa=is_pubmedqa) for r in rows]
    enc = tokenizer(prompts, truncation=True, max_length=max_src_len)
    lab = tokenizer(text_target=targets, truncation=True, max_length=max_tgt_len)
    feats = []
    for i in range(len(rows)):
        t_raw = get_trust(rows[i])
        t_val = float(t_raw) if t_raw is not None else 1.0   # only reachable for dev/Fisher rows
        feats.append({
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": lab["input_ids"][i],
            "trust_score": t_val
        })
    return feats


# Fact-Checked Trust Scores for the Clean Stream
TRUST_FIELD_ALIASES = ("trust_score", "trust", "tau", "tau_x", "trust_weight", "trustScore")

def get_trust(row: Dict[str, Any]) -> Optional[float]:
    """Returns the fact-checking trust score of a row, whatever the field is called."""
    for k in TRUST_FIELD_ALIASES:
        v = row.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None

def _as_text(x: Any) -> str:
    if isinstance(x, (list, tuple)):
        return " ".join(_as_text(v) for v in x)
    if isinstance(x, dict):
        return " ".join(_as_text(v) for v in x.values())
    return str(x or "")

def _norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", _as_text(x)).strip()

def _relaxed_text(x: Any) -> str:
    """Formatting-insensitive form: Unicode-normalised, lower-cased, letters and digits only.
    Whitespace, punctuation, quotes or tokenisation differences disappear, but any change
    of words (negation, reversal, deletion, injected claim) still yields a different string."""
    t = unicodedata.normalize("NFKC", _as_text(x)).lower()
    return re.sub(r"[^0-9a-z]+", "", t)

def sample_content_key(row: Dict[str, Any], is_pubmedqa: bool) -> str:
    """Exact identity of a sample by its content (whitespace-normalised)."""
    q = _norm_text(row.get("question"))
    c = _norm_text(row.get("context"))
    a = normalize_label(row.get("answer"), is_pubmedqa=is_pubmedqa)
    return hashlib.sha1(f"{q}\x1f{c}\x1f{a}".encode("utf-8")).hexdigest()

def sample_relaxed_key(row: Dict[str, Any], is_pubmedqa: bool) -> str:
    """Formatting-insensitive identity of a sample (same words, same label)."""
    q = _relaxed_text(row.get("question"))
    c = _relaxed_text(row.get("context"))
    a = normalize_label(row.get("answer"), is_pubmedqa=is_pubmedqa)
    return hashlib.sha1(f"{q}\x1f{c}\x1f{a}".encode("utf-8")).hexdigest()

def _iter_split_rows(directory: str):
    for f in sorted(os.listdir(directory)):
        if f.startswith("s") and f.endswith(".json"):
            rows = load_json(os.path.join(directory, f))
            if isinstance(rows, list):
                yield f, rows

def build_clean_trust_index(
    base_dir: str,
    is_pubmedqa: bool,
    clean_trust_dir: Optional[str] = None,
    extra_files: Optional[List[str]] = None
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Maps sample content -> trust score produced by the fact-checking module.
    Two maps are built: 'exact' (whitespace-normalised text) and 'relaxed' (letters and
    digits only). Sources, by priority: --clean_trust_dir, --clean_trust_extra, then every
    fact-checked directory under FactCheckedData/ (unpoisoned samples have the clean text)."""
    index: Dict[str, Dict[str, float]] = {"exact": {}, "relaxed": {}}
    conflicts: List[float] = []
    n_rows, n_rows_with_trust = 0, 0
    field_names: Dict[str, int] = {}

    def _add(rows: List[Dict[str, Any]], override: bool):
        nonlocal n_rows, n_rows_with_trust
        for r in rows:
            n_rows += 1
            for k in r.keys():
                field_names[k] = field_names.get(k, 0) + 1
            t = get_trust(r)
            if t is None:
                continue
            n_rows_with_trust += 1
            for kind, key in (("exact", sample_content_key(r, is_pubmedqa)),
                              ("relaxed", sample_relaxed_key(r, is_pubmedqa))):
                m = index[kind]
                if key in m:
                    if abs(m[key] - t) > 1e-6:
                        conflicts.append(abs(m[key] - t))
                    if not override:
                        continue
                m[key] = t

    fc_dir = os.path.join(base_dir, "FactCheckedData")
    if os.path.isdir(fc_dir):
        for d in sorted(os.listdir(fc_dir)):
            full = os.path.join(fc_dir, d)
            if os.path.isdir(full):
                for _, rows in _iter_split_rows(full):
                    _add(rows, override=False)
    for ef in (extra_files or []):
        rows = load_json(ef)
        _add(rows if isinstance(rows, list) else [], override=True)
    if clean_trust_dir and os.path.isdir(clean_trust_dir):
        for _, rows in _iter_split_rows(clean_trust_dir):
            _add(rows, override=True)

    info = {"n_rows_scanned": n_rows, "n_rows_with_trust": n_rows_with_trust,
            "n_indexed": len(index["exact"]), "n_conflicts": len(conflicts),
            "max_conflict_abs_diff": max(conflicts, default=0.0), "field_names": field_names}
    if n_rows and not n_rows_with_trust:
        print(f"[!] None of the {n_rows} fact-checked rows has a trust field "
              f"({', '.join(TRUST_FIELD_ALIASES)}). Fields found: {sorted(field_names)}")
    if conflicts:
        print(f"[!] Warning: {len(conflicts)} samples received different trust scores in different "
              f"files (max |diff| = {info['max_conflict_abs_diff']:.4f}). Higher-priority source kept.")
    return index, info

def attach_clean_trust(
    rows: List[Dict[str, Any]],
    index: Dict[str, Dict[str, float]],
    is_pubmedqa: bool,
    counts: Optional[Dict[str, int]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (rows_with_trust, missing_rows). Exact match first, then formatting-insensitive
    match. Rows are copied; originals are untouched."""
    out, missing = [], []
    for r in rows:
        r2 = {k: v for k, v in r.items() if k not in TRUST_FIELD_ALIASES}
        t = index["exact"].get(sample_content_key(r, is_pubmedqa))
        how = "exact"
        if t is None:
            t = index["relaxed"].get(sample_relaxed_key(r, is_pubmedqa))
            how = "relaxed"
        if t is None:
            missing.append(r)
        else:
            r2["trust_score"] = t
            if counts is not None:
                counts[how] = counts.get(how, 0) + 1
        out.append(r2)
    return out, missing

def _first_difference(a: str, b: str, width: int = 60) -> str:
    n = min(len(a), len(b))
    i = next((k for k in range(n) if a[k] != b[k]), n)
    if i == n and len(a) == len(b):
        return "identical"
    lo = max(0, i - 20)
    return (f"first difference at char {i} (lengths {len(a)} vs {len(b)}):\n"
            f"        clean : ...{a[lo:i + width]!r}\n"
            f"        f-chk : ...{b[lo:i + width]!r}")

def diagnose_clean_trust(dataset_name: str, base_dir: str, max_examples: int = 3) -> None:
    """Explains why clean rows do not match their fact-checked copies: field names, trust
    field, ids, and the first textual difference for a few samples of the first split."""
    is_pubmedqa = (dataset_name.lower() == "pubmedqa")
    split_dir = os.path.join(base_dir, "baselines", "_2_Isolate_Data")
    fc_dir = os.path.join(base_dir, "FactCheckedData")
    print(f"\n{'='*78}\n DIAGNOSIS | {dataset_name}\n{'='*78}")
    if not os.path.isdir(fc_dir):
        print(f"  [!] {fc_dir} does not exist."); return
    fc_versions = [d for d in sorted(os.listdir(fc_dir)) if os.path.isdir(os.path.join(fc_dir, d))]
    print(f"  Fact-checked folders: {fc_versions}")
    clean_files = [f for f, _ in _iter_split_rows(split_dir)]
    if not clean_files or not fc_versions:
        print("  [!] Nothing to compare."); return
    f0 = sorted(clean_files, key=lambda x: int(re.search(r"\d+", x).group()))[0]
    clean_rows = load_json(os.path.join(split_dir, f0))
    fc_path = os.path.join(fc_dir, fc_versions[0], f0)
    if not os.path.exists(fc_path):
        print(f"  [!] {fc_path} not found."); return
    fc_rows = load_json(fc_path)
    print(f"  Comparing {split_dir}/{f0} ({len(clean_rows)} rows) with {fc_versions[0]}/{f0} ({len(fc_rows)} rows)")
    print(f"  Clean fields       : {sorted(clean_rows[0].keys())}")
    print(f"  Fact-checked fields: {sorted(fc_rows[0].keys())}")
    print(f"  Trust value (1st fact-checked row): {get_trust(fc_rows[0])!r}  "
          f"(looked for: {', '.join(TRUST_FIELD_ALIASES)})")
    for fld in ("question", "context", "answer"):
        print(f"  type of '{fld}': clean={type(clean_rows[0].get(fld)).__name__}, "
              f"fact-checked={type(fc_rows[0].get(fld)).__name__}")

    by_id = {str(r.get("id")): r for r in fc_rows if r.get("id") is not None}
    by_q = {_relaxed_text(r.get("question")): r for r in fc_rows}
    shown = 0
    n_exact = n_relaxed = n_paired = 0
    for r in clean_rows:
        partner = by_id.get(str(r.get("id"))) if r.get("id") is not None else None
        if partner is None:
            partner = by_q.get(_relaxed_text(r.get("question")))
        if partner is None:
            continue
        n_paired += 1
        if sample_content_key(r, is_pubmedqa) == sample_content_key(partner, is_pubmedqa):
            n_exact += 1; continue
        if sample_relaxed_key(r, is_pubmedqa) == sample_relaxed_key(partner, is_pubmedqa):
            n_relaxed += 1; continue
        if shown < max_examples:
            shown += 1
            print(f"\n  Example {shown}: id={r.get('id')!r}")
            for fld in ("question", "context"):
                print(f"    {fld}: {_first_difference(_norm_text(r.get(fld)), _norm_text(partner.get(fld)))}")
            print(f"    answer: clean={r.get('answer')!r} -> {normalize_label(r.get('answer'), is_pubmedqa)}, "
                  f"fact-checked={partner.get('answer')!r} -> {normalize_label(partner.get('answer'), is_pubmedqa)}")
    print(f"\n  Paired {n_paired}/{len(clean_rows)} clean rows with a fact-checked row (by id or question): "
          f"{n_exact} exact match, {n_relaxed} formatting-only match, "
          f"{n_paired - n_exact - n_relaxed} different text (expected only for poisoned samples, ~20%).")

def _has_split_files(d: str) -> bool:
    return os.path.isdir(d) and any(f.startswith("s") and f.endswith(".json") for f in os.listdir(d))

def resolve_clean_fc_dir(base_dir: str, override: Optional[str] = None) -> Optional[str]:
    """Locates the fact-checked clean splits: --clean_trust_dir if given, otherwise
    FactCheckedData/Clean_Fact_Checked (split files directly inside, or one level below)."""
    cand = override or os.path.join(base_dir, "FactCheckedData", CLEAN_FC_DIRNAME)
    if _has_split_files(cand):
        return cand
    if os.path.isdir(cand):
        subs = [os.path.join(cand, d) for d in sorted(os.listdir(cand)) if _has_split_files(os.path.join(cand, d))]
        if len(subs) == 1:
            return subs[0]
        if len(subs) > 1:
            print(f"[!] {cand} contains several split folders {subs}; pass the right one with --clean_trust_dir.")
    elif override:
        print(f"[!] --clean_trust_dir {override} does not exist.")
    return None

def load_fact_checked_clean_split(
    clean_fc_dir: str,
    baseline_split_dir: str,
    s_name: str,
    is_pubmedqa: bool
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Loads one fact-checked clean split and checks it against the baseline clean split.
    Returns (rows with 'trust_score', report). Raises if the file or a trust score is missing."""
    fc_path = os.path.join(clean_fc_dir, f"{s_name}.json")
    if not os.path.exists(fc_path):
        raise FileNotFoundError(f"[!] Fact-checked clean split not found: {fc_path}")
    fc_rows = load_json(fc_path)
    rows, no_trust = [], []
    for i, r in enumerate(fc_rows):
        t = get_trust(r)
        if t is None:
            no_trust.append(i)
            continue
        r2 = {k: v for k, v in r.items() if k not in TRUST_FIELD_ALIASES}
        r2["trust_score"] = float(t)
        rows.append(r2)
    if no_trust:
        raise ValueError(f"[!] {len(no_trust)}/{len(fc_rows)} rows in {fc_path} have no trust score "
                         f"(looked for: {', '.join(TRUST_FIELD_ALIASES)}); first indices: {no_trust[:5]}")

    # Consistency with the baseline clean split used by all other methods.
    report = {"n_fact_checked": len(rows), "n_baseline": None, "n_matched": None}
    base_path = os.path.join(baseline_split_dir, f"{s_name}.json")
    if os.path.exists(base_path):
        base_rows = load_json(base_path)
        fc_keys = {sample_relaxed_key(r, is_pubmedqa) for r in rows}
        matched = sum(sample_relaxed_key(r, is_pubmedqa) in fc_keys for r in base_rows)
        report.update({"n_baseline": len(base_rows), "n_matched": matched})
    return rows, report

def check_clean_trust_coverage(
    dataset_name: str,
    base_dir: str,
    index: Dict[str, float],
    report_dir: str,
    clean_fc_dir: Optional[str] = None
) -> bool:
    """Preflight: per-split coverage of fact-checked trust on the clean stream. Exports the
    clean samples that still need fact-checking. Returns True if coverage is complete."""
    is_pubmedqa = (dataset_name.lower() == "pubmedqa")
    split_dir = os.path.join(base_dir, "baselines", "_2_Isolate_Data")
    all_missing: List[Dict[str, Any]] = []
    all_trust: List[float] = []

    if clean_fc_dir:
        print(f"\n[Clean trust | {dataset_name}] source: {clean_fc_dir}")
        ok = True
        for f, _ in sorted(_iter_split_rows(split_dir), key=lambda x: int(re.search(r"\d+", x[0]).group())):
            s_name = f[:-5]
            try:
                rows, rep_ = load_fact_checked_clean_split(clean_fc_dir, split_dir, s_name, is_pubmedqa)
            except (FileNotFoundError, ValueError) as e:
                print(f"  {f:<10} {e}"); ok = False; continue
            taus = [r["trust_score"] for r in rows]
            all_trust += taus
            match = (f"{rep_['n_matched']}/{rep_['n_baseline']} baseline samples matched"
                     if rep_["n_baseline"] is not None else "no baseline file")
            flag = "" if rep_["n_matched"] == rep_["n_baseline"] else "  <-- differs from baseline"
            print(f"  {f:<10} {len(rows):>4} rows | mean tau {np.mean(taus):.3f} | "
                  f"tau>={FULL_EWC_TRUST_THRESHOLD}: {np.mean([t >= FULL_EWC_TRUST_THRESHOLD for t in taus]):.1%} | {match}{flag}")
        if all_trust:
            print(f"  ALL        mean tau {np.mean(all_trust):.3f} | "
                  f"tau>={FULL_EWC_TRUST_THRESHOLD}: {np.mean([t >= FULL_EWC_TRUST_THRESHOLD for t in all_trust]):.1%}")
        print("  [✓] Fact-checked clean splits are complete." if ok else "  [!] Some fact-checked clean splits are missing or incomplete.")
        return ok

    print(f"\n[Clean trust coverage | {dataset_name}] (no {CLEAN_FC_DIRNAME} folder: matching against FactCheckedData)")
    for f, rows in _iter_split_rows(split_dir):
        counts: Dict[str, int] = {}
        with_t, missing = attach_clean_trust(rows, index, is_pubmedqa, counts)
        taus = [r["trust_score"] for r in with_t if "trust_score" in r]
        all_trust += taus
        ge = np.mean([t >= FULL_EWC_TRUST_THRESHOLD for t in taus]) if taus else float("nan")
        print(f"  {f:<10} covered {len(rows) - len(missing):>4}/{len(rows):<4} | "
              f"(exact {counts.get('exact', 0)}, relaxed {counts.get('relaxed', 0)}) | "
              f"mean tau {np.mean(taus) if taus else float('nan'):.3f} | tau>={FULL_EWC_TRUST_THRESHOLD}: {ge:.1%}")
        for r in missing:
            r2 = dict(r); r2["_split"] = f
            all_missing.append(r2)
    if all_trust:
        print(f"  ALL        mean tau {np.mean(all_trust):.3f} | "
              f"tau>={FULL_EWC_TRUST_THRESHOLD}: {np.mean([t >= FULL_EWC_TRUST_THRESHOLD for t in all_trust]):.1%}")
    os.makedirs(report_dir, exist_ok=True)
    miss_path = os.path.join(report_dir, f"clean_samples_missing_trust_{dataset_name}.json")
    save_json(all_missing, miss_path)
    if all_missing:
        print(f"  [!] {len(all_missing)} clean samples have no fact-checked trust score.\n"
              f"      Run them through the fact-checking module and pass the result with\n"
              f"      --clean_trust_extra <file.json> (rows with question/context/answer/trust_score).\n"
              f"      Exported to: {miss_path}")
        return False
    print("  [✓] Every clean training sample has a fact-checked trust score.")
    return True

class MedFactSeq2SeqCollator:
    """Seq2Seq Collator that safely extracts and passes batch trust_scores."""
    def __init__(self, tokenizer: AutoTokenizer, model: Any = None):
        self.base_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, padding="longest", pad_to_multiple_of=8, label_pad_token_id=-100
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        trust_scores = [f["trust_score"] for f in features]
        clean_features = [{k: v for k, v in f.items() if k != "trust_score"} for f in features]
        batch = self.base_collator(clean_features)
        batch["trust_score"] = torch.tensor(trust_scores, dtype=torch.float32)
        return batch

@torch.no_grad()
def compute_dev_loss(model: AutoModelForSeq2SeqLM, dev_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in dev_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        loss = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]).loss
        n_tok = int((batch["labels"] != -100).sum().item())
        if torch.isfinite(loss):
            total_loss += float(loss.item()) * n_tok
            total_tokens += n_tok
    return total_loss / max(1, total_tokens) if total_tokens else float("inf")

def compute_empirical_fisher(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    train_rows: List[Dict[str, Any]],
    is_pubmedqa: bool,
    device: torch.device,
    max_samples: int = 256,
    batch_size: int = 4,
    max_src_len: int = 512,
    max_tgt_len: int = 8,
    normalize: bool = False
) -> Dict[str, torch.Tensor]:
    """Computes empirical diagonal Fisher Information Matrix (FIM) for trainable parameters."""
    model.eval()
    fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}

    sample_rows = train_rows[:max_samples] if len(train_rows) > max_samples else train_rows
    feats = _tokenize_rows_with_trust(tokenizer, sample_rows, is_pubmedqa, max_src_len, max_tgt_len)
    collator = MedFactSeq2SeqCollator(tokenizer, model=model)
    loader = DataLoader(feats, batch_size=batch_size, shuffle=False, collate_fn=collator)

    num_samples = len(sample_rows)
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        if torch.isfinite(loss):
            loss.backward()
            bs = batch["input_ids"].size(0)
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += (p.grad.detach() ** 2) * bs

    model.zero_grad(set_to_none=True)

    for n in fisher:
        fisher[n] = torch.nan_to_num(fisher[n] / max(1, num_samples), nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        tot_sum = sum(f.sum().item() for f in fisher.values())
        tot_num = sum(f.numel() for f in fisher.values())
        mean_val = tot_sum / max(1, tot_num)
        if mean_val > 1e-12:
            for n in fisher:
                fisher[n] = fisher[n] / mean_val

    return fisher

def compute_step_loss(
    model: AutoModelForSeq2SeqLM,
    batch: Dict[str, torch.Tensor],
    cand_t: torch.Tensor,
    margin: float = 0.20,
    margin_lambda: float = 0.35
) -> torch.Tensor:
    """Combines Seq2Seq Cross-Entropy with first-step logit candidate margin separation."""
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
    ce_loss = outputs.loss
    if margin_lambda <= 0.0:
        return ce_loss

    logits = outputs.logits[:, 0, :]
    cand_logits = logits.index_select(1, cand_t)
    
    first_tok_labels = batch["labels"][:, 0]
    eq = (first_tok_labels.unsqueeze(1) == cand_t.unsqueeze(0))
    valid_mask = eq.any(dim=1)
    
    if not valid_mask.any():
        return ce_loss

    gold_indices = eq[valid_mask].float().argmax(dim=1)
    sub_logits = cand_logits[valid_mask]
    gold_logits = sub_logits.gather(1, gold_indices.unsqueeze(1)).squeeze(1)

    mask_comp = torch.ones_like(sub_logits, dtype=torch.bool)
    mask_comp.scatter_(1, gold_indices.unsqueeze(1), False)
    competitor_logits = sub_logits[mask_comp].view(sub_logits.size(0), -1).max(dim=1)[0]

    margin_loss = F.relu(competitor_logits - gold_logits + margin).mean()
    return ce_loss + margin_lambda * margin_loss


# Single Split Fine-Tuning Step with MedFact-EWC Trust Modulation
def train_single_split_medfact_ewc(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    train_rows: List[Dict[str, Any]],
    dev_rows: List[Dict[str, Any]],
    is_pubmedqa: bool,
    out_split_dir: str,
    hparams: Dict[str, Any],
    anchor_weights: Optional[Dict[str, torch.Tensor]] = None,
    fisher_matrix: Optional[Dict[str, torch.Tensor]] = None,
    seed: int = 42,
    micro_batch_size: int = 4,
    eval_batch_size: int = 16,
    max_tgt_len: int = 8,
    max_grad_norm: float = 1.0
) -> Tuple[AutoModelForSeq2SeqLM, float, Dict[str, float]]:
    hp = {**OPTIMAL_HPARAMS, **hparams}
    device = next(model.parameters()).device
    seed_all(seed)
    os.makedirs(out_split_dir, exist_ok=True)
    shutil.rmtree(os.path.join(out_split_dir, "_temp_run"), ignore_errors=True)

    max_src_len = int(hp["max_src_len"])
    train_feats = _tokenize_rows_with_trust(tokenizer, train_rows, is_pubmedqa, max_src_len, max_tgt_len,
                                            require_trust=True)
    # Dev rows are only used for the dev loss (trust is not used there).
    dev_feats = _tokenize_rows_with_trust(tokenizer, dev_rows, is_pubmedqa, max_src_len, max_tgt_len)

    collator = MedFactSeq2SeqCollator(tokenizer, model=model)
    batch_size = int(hp["per_device_train_batch_size"])
    micro_bs = max(1, min(batch_size, micro_batch_size))
    grad_accum = max(1, batch_size // micro_bs)

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(train_feats, batch_size=micro_bs, shuffle=True, collate_fn=collator, generator=gen, pin_memory=True)
    dev_loader = DataLoader(dev_feats, batch_size=eval_batch_size, shuffle=False, collate_fn=collator, pin_memory=True)

    epochs = int(hp["num_train_epochs"])
    n_micro = len(train_loader)
    steps_per_epoch = math.ceil(n_micro / grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = math.ceil(float(hp["warmup_ratio"]) * total_steps)

    cand_names, cand_token_ids = get_candidate_info(tokenizer, is_pubmedqa)
    cand_t = torch.tensor(cand_token_ids, dtype=torch.long, device=device)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if ("bias" in n or "layer_norm" in n) else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(hp["weight_decay"])},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(hp["learning_rate"]), betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ewc_lambda = float(hp["ewc_lambda"])
    has_ewc = (anchor_weights is not None and fisher_matrix is not None and ewc_lambda > 0.0)

    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    skipped_steps = 0
    patience = int(hp["patience"])
    overall_mean_trust: float = 1.0
    total_batches, full_ewc_batches = 0, 0
    sample_trust = [float(get_trust(r)) for r in train_rows]
    sample_trust_ge_thr = float(np.mean([t >= FULL_EWC_TRUST_THRESHOLD for t in sample_trust])) if sample_trust else 1.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        t_ep = time.time()
        run_task_loss, run_ewc_loss, run_total_loss, run_n = 0.0, 0.0, 0.0, 0
        run_trust_sum = 0.0
        group_nonfinite = False

        desc_label = f"  epoch {epoch}/{epochs} (MedFact-EWC)" if has_ewc else f"  epoch {epoch}/{epochs}"
        pbar = tqdm(train_loader, total=n_micro, leave=False, dynamic_ncols=True, desc=desc_label)
        for step, batch in enumerate(pbar):
            group_idx = step // grad_accum
            group_len = min(grad_accum, n_micro - group_idx * grad_accum)

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            task_loss = compute_step_loss(model, batch, cand_t, margin=float(hp["semantic_margin"]), margin_lambda=float(hp["margin_lambda"]))

            # Extract mini-batch stability coefficient tau_bar
            tau_bar = float(batch["trust_score"].mean().item())
            run_trust_sum += tau_bar
            total_batches += 1
            if tau_bar >= FULL_EWC_TRUST_THRESHOLD:
                full_ewc_batches += 1

            if not torch.isfinite(task_loss):
                group_nonfinite = True
            else:
                # EWC Quadratic Penalty: (lambda_ewc / 2) * sum_j [ F_j * (theta_j - theta*_j)^2 ]
                if has_ewc:
                    ewc_penalty = torch.tensor(0.0, device=device)
                    for n, p in model.named_parameters():
                        if p.requires_grad and n in fisher_matrix and n in anchor_weights:
                            diff = p - anchor_weights[n]
                            ewc_penalty = ewc_penalty + (fisher_matrix[n] * diff.pow(2)).sum()
                    ewc_loss = (ewc_lambda / 2.0) * ewc_penalty

                    # MedFact-EWC Trust Modulation:
                    # On poisoned/unreliable batches (tau_bar < 0.999): tau_bar * L_task + (1 - tau_bar) * L_reg
                    # On clean batches (tau_bar >= 0.999): full EWC (L_task + L_reg)
                    if tau_bar < FULL_EWC_TRUST_THRESHOLD:
                        total_loss = (tau_bar * task_loss) + ((1.0 - tau_bar) * ewc_loss)
                    else:
                        total_loss = task_loss + ewc_loss
                else:
                    ewc_loss = torch.tensor(0.0, device=device)
                    total_loss = task_loss

                (total_loss / group_len).backward()

                run_task_loss += float(task_loss.item())
                run_ewc_loss += float(ewc_loss.item())
                run_total_loss += float(total_loss.item())
                run_n += 1

            if (step + 1) % grad_accum == 0 or (step + 1) == n_micro:
                if group_nonfinite:
                    skipped_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    group_nonfinite = False
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                avg_t_loss = run_task_loss / max(1, run_n)
                avg_e_loss = run_ewc_loss / max(1, run_n)
                avg_tot_loss = run_total_loss / max(1, run_n)
                avg_t_score = run_trust_sum / max(1, run_n)
                if has_ewc:
                    pbar.set_postfix(tot=f"{avg_tot_loss:.4f}", task=f"{avg_t_loss:.4f}", ewc=f"{avg_e_loss:.4f}", tau=f"{avg_t_score:.2f}")
                else:
                    pbar.set_postfix(loss=f"{avg_tot_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        pbar.close()

        # Dev evaluation for early stopping (evaluates task generalisation)
        dev_loss = compute_dev_loss(model, dev_loader, device)
        train_task_loss = run_task_loss / max(1, run_n)
        train_ewc_loss = run_ewc_loss / max(1, run_n)
        train_total_loss = run_total_loss / max(1, run_n)
        overall_mean_trust = run_trust_sum / max(1, run_n)

        improved = dev_loss < best_loss
        if improved:
            best_loss, best_epoch, bad_epochs = dev_loss, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        history.append({
            "epoch": epoch, 
            "train_total_loss": round(train_total_loss, 5),
            "train_task_loss": round(train_task_loss, 5), 
            "train_ewc_loss": round(train_ewc_loss, 5),
            "dev_loss": round(dev_loss, 5),
            "mean_batch_trust": round(overall_mean_trust, 4),
            "improved": improved, 
            "peak_vram_gb": round(peak_gb, 2), 
            "seconds": round(time.time() - t_ep, 1)
        })
        ewc_str = f" | ewc_loss {train_ewc_loss:.4f} | tau {overall_mean_trust:.2f}" if has_ewc else ""
        print(f"  epoch {epoch}/{epochs} | total_loss {train_total_loss:.4f} | task_loss {train_task_loss:.4f}{ewc_str} | dev_loss {dev_loss:.4f}"
              f"{'  *best*' if improved else ''} | peak VRAM {peak_gb:.2f} GB | {time.time() - t_ep:.1f}s", flush=True)

        if bad_epochs >= patience:
            print(f"  [early stop] no dev improvement for {patience} epochs -> restoring epoch {best_epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_json({
        "best_epoch": best_epoch, 
        "best_dev_loss": round(best_loss, 5),
        "ewc_lambda": ewc_lambda if has_ewc else 0.0,
        "mean_split_trust": round(overall_mean_trust, 4),
        "full_ewc_batch_frac": round(full_ewc_batches / max(1, total_batches), 4),
        "sample_trust_ge_thr_frac": round(sample_trust_ge_thr, 4),
        "skipped_nonfinite_steps": skipped_steps, 
        "epochs": history
    }, os.path.join(out_split_dir, "training_log.json"))

    del optimizer, scheduler, best_state, train_loader, dev_loader, collator, train_feats, dev_feats
    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    trust_stats = {
        "full_ewc_batch_frac": round(full_ewc_batches / max(1, total_batches), 4),
        "sample_trust_ge_thr_frac": round(sample_trust_ge_thr, 4),
    }
    return model, overall_mean_trust, trust_stats


# Continual Learning Runners with MedFact-EWC State Management
def run_clean_baseline_seed(
    dataset_name: str,
    seed: int,
    base_dir: str,
    out_dir_seed: str,
    live_csv: str,
    model_name: str,
    device: torch.device,
    hparams: Dict[str, Any],
    trust_index: Dict[str, float],
    clean_fc_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Clean MedFact-EWC run. Every clean sample carries the trust score the fact-checking
    module assigned to its (unperturbed) text, so the clean run is the true within-method
    counterpart of the poisoned run."""
    is_pubmedqa = (dataset_name.lower() == "pubmedqa")
    hp = {**OPTIMAL_HPARAMS, **hparams}

    split_dir = os.path.join(base_dir, "baselines", "_2_Isolate_Data")
    dev_rows = load_json(os.path.join(base_dir, "baselines", "_0_fixed_test", "dev.json"))
    test_rows = load_json(os.path.join(base_dir, "baselines", "_0_fixed_test", "test.json"))

    split_files = sorted([f for f in os.listdir(split_dir) if f.startswith("s") and f.endswith(".json")],
                         key=lambda x: int(re.search(r'\d+', x).group()))
    total_splits = len(split_files)

    summary_file = os.path.join(out_dir_seed, "all_splits_metrics.json")
    rolling_ckpt_dir = os.path.join(out_dir_seed, "_last_checkpoint")

    if os.path.exists(summary_file):
        try:
            saved = load_json(summary_file)
            if len(saved) == total_splits and all(is_valid_metrics_file(os.path.join(out_dir_seed, f"s{i}", "metrics.json")) for i in range(1, total_splits + 1)):
                print(f"[✓ Skipped] Clean Baseline for {dataset_name} (Seed {seed}) is already complete.")
                return saved
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    completed_splits: Dict[str, Any] = {}
    last_completed_idx = 0
    for s_idx in range(1, total_splits + 1):
        mf = os.path.join(out_dir_seed, f"s{s_idx}", "metrics.json")
        if is_valid_metrics_file(mf):
            completed_splits[f"s{s_idx}"] = load_json(mf)
            last_completed_idx = s_idx
        else:
            break

    ckpt_path, ckpt_split = resolve_rolling_checkpoint(rolling_ckpt_dir, legacy_split=last_completed_idx)
    anchor_weights: Optional[Dict[str, torch.Tensor]] = None
    fisher_matrix: Optional[Dict[str, torch.Tensor]] = None

    if ckpt_path and 1 <= ckpt_split <= last_completed_idx:
        for s_i in range(ckpt_split + 1, last_completed_idx + 1):
            completed_splits.pop(f"s{s_i}", None)
        print(f"\n[⚡ RESUMING] Clean {dataset_name} (Seed {seed}): Resuming at split s{ckpt_split + 1}...")
        current_model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).to(device)
        anchor_weights, fisher_matrix = load_ewc_state(ckpt_path, device)
        start_split_idx = ckpt_split + 1
    else:
        current_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        start_split_idx = 1

    all_metrics = copy.deepcopy(completed_splits)

    for s_idx in range(start_split_idx, total_splits + 1):
        s_name = f"s{s_idx}"
        split_out = os.path.join(out_dir_seed, s_name)
        active_ewc_lambda = float(hp["ewc_lambda"]) if (anchor_weights is not None and fisher_matrix is not None) else 0.0
        print(f"\n[Seed {seed} | Clean (fact-checked) {dataset_name}] Training Split {s_idx}/{total_splits} ({s_name}.json) [MedFact-EWC λ={active_ewc_lambda:.0f}]...")
        if clean_fc_dir:
            # Primary source: the clean split as passed through the fact-checking module.
            train_rows, fc_report = load_fact_checked_clean_split(clean_fc_dir, split_dir, s_name, is_pubmedqa)
            if fc_report["n_baseline"] is not None and fc_report["n_matched"] != fc_report["n_baseline"]:
                print(f"  [!] Warning: only {fc_report['n_matched']}/{fc_report['n_baseline']} baseline samples of {s_name} "
                      f"appear in {CLEAN_FC_DIRNAME} ({fc_report['n_fact_checked']} rows). The clean run trains on the "
                      f"fact-checked file; check that it contains the same samples as the baseline split.")
            raw_rows, missing = train_rows, []
        else:
            raw_rows = load_json(os.path.join(split_dir, f"{s_name}.json"))
            train_rows, missing = attach_clean_trust(raw_rows, trust_index, is_pubmedqa)
        if missing:
            raise ValueError(
                f"[!] {len(missing)}/{len(raw_rows)} clean samples in {s_name} have no fact-checked trust score.\n"
                f"    Run with --check_clean_trust to see per-split coverage, or --diagnose_clean_trust to see why\n"
                f"    clean rows do not match their fact-checked copies. Samples that were never fact-checked in\n"
                f"    clean form can be added with --clean_trust_extra."
            )

        t0 = time.time()
        current_model, mean_trust, trust_stats = train_single_split_medfact_ewc(
            model=current_model, tokenizer=tokenizer, train_rows=train_rows, dev_rows=dev_rows,
            is_pubmedqa=is_pubmedqa, out_split_dir=split_out, hparams=hp,
            anchor_weights=anchor_weights, fisher_matrix=fisher_matrix, seed=seed
        )
        eval_metrics, predictions_log = evaluate_fixed_test(
            model=current_model, tokenizer=tokenizer, test_rows=test_rows, is_pubmedqa=is_pubmedqa,
            device=device, max_src_len=int(hp["max_src_len"])
        )
        duration = time.time() - t0

        # Compute empirical Fisher on completed split and update Online EWC state
        print(f"  [⚡ MedFact-EWC] Estimating empirical Fisher on split {s_name} ({len(train_rows)} samples)...")
        t_fish = time.time()
        new_fisher = compute_empirical_fisher(
            model=current_model, tokenizer=tokenizer, train_rows=train_rows, is_pubmedqa=is_pubmedqa,
            device=device, max_samples=int(hp["fisher_max_samples"]),
            max_src_len=int(hp["max_src_len"]), normalize=bool(hp["fisher_normalize"])
        )
        gamma = float(hp["fisher_gamma"])
        if fisher_matrix is None:
            fisher_matrix = new_fisher
        else:
            for n in new_fisher:
                fisher_matrix[n] = gamma * fisher_matrix[n] + new_fisher[n]

        anchor_weights = {n: p.detach().clone() for n, p in current_model.named_parameters() if p.requires_grad}
        print(f"  [⚡ MedFact-EWC] Fisher consolidated (gamma={gamma}) in {time.time() - t_fish:.1f}s")

        eval_metrics["avg_trust"] = round(mean_trust, 4)
        eval_metrics.update(trust_stats)
        all_metrics[s_name] = eval_metrics
        save_json(eval_metrics, os.path.join(split_out, "metrics.json"))
        save_json(predictions_log, os.path.join(split_out, "predictions.json"))
        save_rolling_checkpoint(current_model, tokenizer, rolling_ckpt_dir, s_idx, anchor_weights, fisher_matrix)

        print_metric_box(f"Split {s_name} Clean (fact-checked) Test Results (Seed {seed} | {dataset_name})", eval_metrics, duration, active_ewc_lambda, mean_trust)
        append_live_split_csv(live_csv, {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": seed, "dataset": dataset_name, "setting": "clean_factchecked", "technique": "none",
            "version": "v0", "split": s_name, "total_splits": total_splits,
            "accuracy": eval_metrics["accuracy"], "f1_macro": eval_metrics["f1_macro"],
            "precision_macro": eval_metrics["precision_macro"], "recall_macro": eval_metrics["recall_macro"],
            "auroc": eval_metrics.get("auroc", ""), "brier_score": eval_metrics.get("brier_score", ""),
            "ewc_lambda": active_ewc_lambda, "avg_trust": round(mean_trust, 4), "duration_sec": round(duration, 1),
            **trust_stats
        })
        save_json(all_metrics, summary_file)

    del current_model, anchor_weights, fisher_matrix
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_metrics

def run_poisoned_versions_seed(
    dataset_name: str,
    technique: str,
    seed: int,
    base_dir: str,
    out_dir_tech_seed: str,
    live_csv: str,
    model_name: str,
    device: torch.device,
    hparams: Dict[str, Any],
    versions: Optional[List[int]] = None,
    keep_snapshots: bool = False
) -> Dict[str, Dict[str, Any]]:
    is_pubmedqa = (dataset_name.lower() == "pubmedqa")
    hp = {**OPTIMAL_HPARAMS, **hparams}

    fact_checked_dir = os.path.join(base_dir, "FactCheckedData")
    if not os.path.exists(fact_checked_dir):
        raise FileNotFoundError(f"[!] Critical: FactCheckedData not found in {base_dir}. MedFact-EWC strictly requires precomputed trust scores.")

    dev_rows = load_json(os.path.join(base_dir, "baselines", "_0_fixed_test", "dev.json"))
    test_rows = load_json(os.path.join(base_dir, "baselines", "_0_fixed_test", "test.json"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    snapshots_dir = os.path.join(out_dir_tech_seed, "_poisoned_snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    v_dirs = [d for d in os.listdir(fact_checked_dir) if d.startswith(f"_2_Isolate_Data_{technique}_v")]
    v_nums = sorted([int(re.search(r'v(\d+)', d).group(1)) for d in v_dirs])
    target_versions = [v for v in v_nums if v in versions] if versions else v_nums

    master_summary: Dict[str, Dict[str, Any]] = {}

    for k in target_versions:
        v_src_dir = os.path.join(fact_checked_dir, f"_2_Isolate_Data_{technique}_v{k}")
        v_out_dir = os.path.join(out_dir_tech_seed, f"v{k}")
        os.makedirs(v_out_dir, exist_ok=True)

        split_files = sorted([f for f in os.listdir(v_src_dir) if f.startswith("s") and f.endswith(".json")],
                             key=lambda x: int(re.search(r'\d+', x).group()))
        total_splits = len(split_files)

        summary_file = os.path.join(v_out_dir, "all_splits_metrics.json")
        rolling_v_ckpt_dir = os.path.join(v_out_dir, "_last_checkpoint")

        if os.path.exists(summary_file):
            try:
                saved = load_json(summary_file)
                if len(saved) == total_splits and all(is_valid_metrics_file(os.path.join(v_out_dir, f"s{i}", "metrics.json")) for i in range(1, total_splits + 1)):
                    print(f"[✓ Skipped] Version v{k} for {technique} (Seed {seed}) is already complete.")
                    master_summary[f"v{k}"] = saved
                    continue
            except Exception:
                pass

        completed_splits: Dict[str, Any] = {}
        last_completed_idx = 0
        for s_idx in range(1, total_splits + 1):
            mf = os.path.join(v_out_dir, f"s{s_idx}", "metrics.json")
            if is_valid_metrics_file(mf):
                completed_splits[f"s{s_idx}"] = load_json(mf)
                last_completed_idx = s_idx
            else:
                break

        current_model = None
        anchor_weights: Optional[Dict[str, torch.Tensor]] = None
        fisher_matrix: Optional[Dict[str, torch.Tensor]] = None
        start_split_idx = 1
        snap_k_dir = os.path.join(snapshots_dir, f"snapshot_s{k}")

        if last_completed_idx > 0:
            ckpt_path, ckpt_split = resolve_rolling_checkpoint(rolling_v_ckpt_dir, legacy_split=last_completed_idx)
            snap_k_path, _ = resolve_rolling_checkpoint(snap_k_dir, legacy_split=k)
            if ckpt_path and 1 <= ckpt_split <= last_completed_idx:
                for s_i in range(ckpt_split + 1, last_completed_idx + 1):
                    completed_splits.pop(f"s{s_i}", None)
                print(f"\n[⚡ RESUMING] Version v{k} (Seed {seed}): Resuming at split s{ckpt_split + 1}...")
                current_model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).to(device)
                anchor_weights, fisher_matrix = load_ewc_state(ckpt_path, device)
                start_split_idx = ckpt_split + 1
            elif last_completed_idx >= k and snap_k_path:
                for s_i in range(k + 1, last_completed_idx + 1):
                    completed_splits.pop(f"s{s_i}", None)
                print(f"\n[⚡ RESUMING] Version v{k} (Seed {seed}): Boundary split s{k} complete. Resuming from snapshot...")
                current_model = AutoModelForSeq2SeqLM.from_pretrained(snap_k_path).to(device)
                anchor_weights, fisher_matrix = load_ewc_state(snap_k_path, device)
                start_split_idx = k + 1

        if current_model is None:
            if k > 1:
                prev_summary = os.path.join(out_dir_tech_seed, f"v{k-1}", "all_splits_metrics.json")
                prev_snap, _ = resolve_rolling_checkpoint(os.path.join(snapshots_dir, f"snapshot_s{k-1}"), legacy_split=k - 1)
                prev_m = load_json(prev_summary) if os.path.exists(prev_summary) else {}
                if prev_snap is not None and all(f"s{i}" in prev_m for i in range(1, k)):
                    print(f"[*] [Seed {seed} | v{k}] Inheriting splits s1..s{k-1} from v{k-1}...")
                    for s_i in range(1, k):
                        s_key = f"s{s_i}"
                        completed_splits[s_key] = prev_m[s_key]
                        save_json(prev_m[s_key], os.path.join(v_out_dir, s_key, "metrics.json"))
                        prev_p = os.path.join(out_dir_tech_seed, f"v{k-1}", s_key, "predictions.json")
                        if os.path.exists(prev_p):
                            shutil.copyfile(prev_p, os.path.join(v_out_dir, s_key, "predictions.json"))
                    current_model = AutoModelForSeq2SeqLM.from_pretrained(prev_snap).to(device)
                    anchor_weights, fisher_matrix = load_ewc_state(prev_snap, device)
                    start_split_idx = k
                else:
                    current_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
                    anchor_weights, fisher_matrix = None, None
                    start_split_idx = 1
            else:
                current_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
                anchor_weights, fisher_matrix = None, None
                start_split_idx = 1

        v_metrics = copy.deepcopy(completed_splits)

        for s_idx in range(start_split_idx, total_splits + 1):
            s_name = f"s{s_idx}"
            split_out = os.path.join(v_out_dir, s_name)
            split_type_str = "POISONED" if s_idx <= k else "CLEAN"
            active_ewc_lambda = float(hp["ewc_lambda"]) if (anchor_weights is not None and fisher_matrix is not None) else 0.0
            print(f"\n>>> [Seed {seed} | {dataset_name} | {technique} | v{k}] Split {s_idx}/{total_splits} ({split_type_str}) [MedFact-EWC λ={active_ewc_lambda:.0f}] <<<")

            train_rows = load_json(os.path.join(v_src_dir, f"{s_name}.json"))
            t0 = time.time()
            current_model, mean_trust, trust_stats = train_single_split_medfact_ewc(
                model=current_model, tokenizer=tokenizer, train_rows=train_rows, dev_rows=dev_rows,
                is_pubmedqa=is_pubmedqa, out_split_dir=split_out, hparams=hp,
                anchor_weights=anchor_weights, fisher_matrix=fisher_matrix, seed=seed
            )
            eval_metrics, predictions_log = evaluate_fixed_test(
                model=current_model, tokenizer=tokenizer, test_rows=test_rows, is_pubmedqa=is_pubmedqa,
                device=device, max_src_len=int(hp["max_src_len"])
            )
            duration = time.time() - t0

            # Compute empirical Fisher on completed split and update Online EWC state
            print(f"  [⚡ MedFact-EWC] Estimating empirical Fisher on split {s_name} ({len(train_rows)} samples)...")
            t_fish = time.time()
            new_fisher = compute_empirical_fisher(
                model=current_model, tokenizer=tokenizer, train_rows=train_rows, is_pubmedqa=is_pubmedqa,
                device=device, max_samples=int(hp["fisher_max_samples"]),
                max_src_len=int(hp["max_src_len"]), normalize=bool(hp["fisher_normalize"])
            )
            gamma = float(hp["fisher_gamma"])
            if fisher_matrix is None:
                fisher_matrix = new_fisher
            else:
                for n in new_fisher:
                    fisher_matrix[n] = gamma * fisher_matrix[n] + new_fisher[n]

            anchor_weights = {n: p.detach().clone() for n, p in current_model.named_parameters() if p.requires_grad}
            print(f"  [⚡ MedFact-EWC] Fisher consolidated (gamma={gamma}) in {time.time() - t_fish:.1f}s")

            if s_idx == k:
                print(f"  [💾] Saving continuation snapshot with EWC state for split s{k} -> {snap_k_dir}")
                save_rolling_checkpoint(current_model, tokenizer, snap_k_dir, k, anchor_weights, fisher_matrix)

            eval_metrics["avg_trust"] = round(mean_trust, 4)
            eval_metrics.update(trust_stats)
            v_metrics[s_name] = eval_metrics
            save_json(eval_metrics, os.path.join(split_out, "metrics.json"))
            save_json(predictions_log, os.path.join(split_out, "predictions.json"))
            save_rolling_checkpoint(current_model, tokenizer, rolling_v_ckpt_dir, s_idx, anchor_weights, fisher_matrix)

            print_metric_box(f"Split {s_name} [{split_type_str}] (Seed {seed} | {dataset_name} | {technique} | v{k})", eval_metrics, duration, active_ewc_lambda, mean_trust)
            append_live_split_csv(live_csv, {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seed": seed, "dataset": dataset_name, "setting": "poisoned", "technique": technique,
                "version": f"v{k}", "split": s_name, "total_splits": total_splits,
                "accuracy": eval_metrics["accuracy"], "f1_macro": eval_metrics["f1_macro"],
                "precision_macro": eval_metrics["precision_macro"], "recall_macro": eval_metrics["recall_macro"],
                "auroc": eval_metrics.get("auroc", ""), "brier_score": eval_metrics.get("brier_score", ""),
                "ewc_lambda": active_ewc_lambda, "avg_trust": round(mean_trust, 4), "duration_sec": round(duration, 1),
                **trust_stats
            })
            save_json(v_metrics, summary_file)

        shutil.rmtree(rolling_v_ckpt_dir, ignore_errors=True)
        master_summary[f"v{k}"] = v_metrics
        del current_model, anchor_weights, fisher_matrix
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return master_summary


# Statistical Aggregations (Mean +/- Std Across Seeds)
def compute_multiseed_aggregations(output_root: str, dataset_name: str, seeds: List[int],
                                   clean_subdir: str = CLEAN_FACTCHECKED_SUBDIR):
    print(f"\n{'='*75}\n COMPUTING MULTI-SEED STATISTICAL AGGREGATIONS: {dataset_name}\n{'='*75}")
    ds_root = os.path.join(output_root, dataset_name)
    num_splits = 12 if dataset_name.lower() == "bioasq2025" else 6

    clean_runs: Dict[int, Dict[str, Any]] = {}
    poison_runs: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {t: {} for t in TECHNIQUES}

    for s in seeds:
        cf = os.path.join(ds_root, clean_subdir, f"seed_{s}", "all_splits_metrics.json")
        if os.path.exists(cf):
            clean_runs[s] = load_json(cf)

    for t in TECHNIQUES:
        t_dir = os.path.join(ds_root, t)
        if not os.path.exists(t_dir):
            continue
        for s in seeds:
            s_dir = os.path.join(t_dir, f"seed_{s}")
            if not os.path.exists(s_dir):
                continue
            for v_name in os.listdir(s_dir):
                vf = os.path.join(s_dir, v_name, "all_splits_metrics.json")
                if v_name.startswith("v") and os.path.exists(vf):
                    poison_runs[t].setdefault(v_name, {})[s] = load_json(vf)

    # 1. Split-Level Aggregation Table
    split_agg_rows = []
    def aggregate_split(setting: str, tech: str, ver: str, s_name: str, runs: Dict[int, Dict[str, Any]]):
        metric_lists: Dict[str, List[float]] = {k: [] for k in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "auroc", "brier_score"]}
        for split_map in runs.values():
            if s_name in split_map:
                for k in metric_lists:
                    if split_map[s_name].get(k) is not None:
                        metric_lists[k].append(float(split_map[s_name][k]))
        row = {"Dataset": dataset_name, "Setting": setting, "Technique": tech, "Version": ver, "Split": s_name, "Seeds_Count": len(metric_lists["f1_macro"])}
        for k, vals in metric_lists.items():
            if vals:
                mu = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                row[f"{k}_mean"], row[f"{k}_std"] = round(mu, 4), round(sd, 4)
                row[f"{k}_display"] = f"{mu*100:.2f} ± {sd*100:.2f}%"
            else:
                row[f"{k}_mean"], row[f"{k}_std"], row[f"{k}_display"] = None, None, "N/A"
        return row

    if clean_runs:
        for i in range(1, num_splits + 1):
            split_agg_rows.append(aggregate_split("Clean_FactChecked", "None", "v0", f"s{i}", clean_runs))
    for t in TECHNIQUES:
        for v_name, s_dict in poison_runs[t].items():
            for i in range(1, num_splits + 1):
                split_agg_rows.append(aggregate_split("Poisoned", t, v_name, f"s{i}", s_dict))

    if split_agg_rows:
        p = os.path.join(ds_root, f"split_metrics_aggregated_mean_std_{dataset_name}.csv")
        pd.DataFrame(split_agg_rows).to_csv(p, index=False, encoding="utf-8")
        print(f"[✓] Saved Split-Level Aggregation: {p}")

    # 2. Overall Summary Across Splits
    overall_rows = []
    def aggregate_overall(setting: str, tech: str, ver: str, runs: Dict[int, Dict[str, Any]]):
        f1s, accs, aurocs = [], [], []
        for split_map in runs.values():
            f1_sub = [m["f1_macro"] for m in split_map.values() if "f1_macro" in m]
            acc_sub = [m["accuracy"] for m in split_map.values() if "accuracy" in m]
            auc_sub = [m["auroc"] for m in split_map.values() if m.get("auroc") is not None]
            if f1_sub: f1s.append(float(np.mean(f1_sub)))
            if acc_sub: accs.append(float(np.mean(acc_sub)))
            if auc_sub: aurocs.append(float(np.mean(auc_sub)))
        if not f1s:
            return None
        f1_m, f1_s = float(np.mean(f1s)), (float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0)
        acc_m, acc_s = float(np.mean(accs)), (float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0)
        auc_m, auc_s = (float(np.mean(aurocs)), float(np.std(aurocs, ddof=1))) if aurocs else (0.0, 0.0)
        return {
            "Dataset": dataset_name, "Setting": setting, "Technique": tech, "Version": ver,
            "Seeds_Count": len(f1s), "Avg_F1_Macro_Mean": round(f1_m, 4), "Avg_F1_Macro_Std": round(f1_s, 4),
            "Avg_F1_Display": f"{f1_m*100:.2f} ± {f1_s*100:.2f}%", "Avg_Accuracy_Mean": round(acc_m, 4),
            "Avg_Accuracy_Std": round(acc_s, 4), "Avg_Accuracy_Display": f"{acc_m*100:.2f} ± {acc_s*100:.2f}%",
            "Avg_AUROC_Mean": round(auc_m, 4), "Avg_AUROC_Std": round(auc_s, 4),
            "Avg_AUROC_Display": f"{auc_m*100:.2f} ± {auc_s*100:.2f}%" if aurocs else "N/A"
        }

    if clean_runs:
        r = aggregate_overall("Clean_FactChecked", "None", "v0", clean_runs)
        if r: overall_rows.append(r)
    for t in TECHNIQUES:
        for v_name, s_dict in poison_runs[t].items():
            r = aggregate_overall("Poisoned", t, v_name, s_dict)
            if r: overall_rows.append(r)

    if overall_rows:
        p = os.path.join(ds_root, f"overall_summary_across_splits_mean_std_{dataset_name}.csv")
        pd.DataFrame(overall_rows).to_csv(p, index=False, encoding="utf-8")
        print(f"[✓] Saved Overall Summary: {p}")

    # 3. Tail Degradation Table
    if clean_runs:
        final_v_name = f"v{num_splits}"
        degrad_rows = []
        for t in TECHNIQUES:
            v_dict = poison_runs[t]
            if not v_dict:
                continue
            avail_vs = sorted(v_dict.keys(), key=lambda x: int(x[1:]))
            target_v = final_v_name if final_v_name in v_dict else avail_vs[-1]
            seed_poison_data = v_dict[target_v]
            row = {"Poisoning_Strategy": t.replace("_", " ").title()}

            for s_idx in range(1, num_splits + 1):
                s_name = f"s{s_idx}"
                drop_list = [clean_runs[s][s_name]["f1_macro"] - seed_poison_data[s][s_name]["f1_macro"]
                             for s in seeds if s in clean_runs and s in seed_poison_data
                             and s_name in clean_runs[s] and s_name in seed_poison_data[s]]
                if drop_list:
                    deg_mean = float(np.mean(drop_list))
                    deg_std = float(np.std(drop_list, ddof=1)) if len(drop_list) > 1 else 0.0
                    row[s_name] = f"{deg_mean:.4f} ± {deg_std:.4f}".replace(".", ",")
                else:
                    row[s_name] = "0,0000 ± 0,0000"
            degrad_rows.append(row)

        if degrad_rows:
            p = os.path.join(ds_root, f"Tail_Degradation_mean_std_{dataset_name}.csv")
            header = ["Poisoning_Strategy"] + [f"s{i}" for i in range(1, num_splits + 1)]
            lines = [";".join(header)] + [";".join([str(r.get(h, "")) for h in header]) for r in degrad_rows]
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print(f"[✓] Saved Tail Degradation Table: {p}")


# Self-Healing Supervisor Process
def _count_completed_splits(output_root: str) -> int:
    n = 0
    for _root, _dirs, files in os.walk(output_root):
        if "metrics.json" in files:
            n += 1
    return n

def run_supervisor(args) -> int:
    # Absolute path: the worker must be found whatever the current working directory is.
    cmd = [sys.executable, "-u", os.path.abspath(__file__)] + sys.argv[1:] + ["--_worker"]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    os.makedirs(args.output_root, exist_ok=True)
    log_path = os.path.join(args.output_root, "supervisor_medfact_ewc_log.txt")
    restarts, no_progress = 0, 0
    last_progress = _count_completed_splits(args.output_root)

    def _log(msg: str):
        line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(f"\n[SUPERVISOR] {msg}", flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    _log(f"Starting MedFact-EWC worker supervisor (completed splits so far: {last_progress})")
    while True:
        # The worker's output is relayed through this process so it is visible in every console
        # (plain terminals, but also Spyder / Jupyter / IDE run panels, which do not show the
        # output of child processes). Raw chunks are forwarded so tqdm progress bars still work.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(1)
                if not chunk:
                    break
                sys.stdout.write(decoder.decode(chunk))
                sys.stdout.flush()
            rc = proc.wait()
        except KeyboardInterrupt:
            _log("Ctrl+C received -> terminating worker process.")
            try:
                proc.terminate()
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
            return 130

        if rc == 0:
            _log("MedFact-EWC Worker completed all splits successfully.")
            return 0

        progress = _count_completed_splits(args.output_root)
        no_progress = 0 if progress > last_progress else no_progress + 1
        last_progress = progress
        restarts += 1

        if no_progress >= 3:
            _log(f"Worker crashed 3 consecutive times without completing any splits. Aborting execution.")
            return rc
        if restarts > args.max_restarts:
            _log(f"Reached --max_restarts={args.max_restarts}. Aborting.")
            return rc

        _log(f"Worker fault detected (code {rc}). Retrying split from last committed state in 20s (attempt {restarts}/{args.max_restarts})...")
        time.sleep(20)


# Main Orchestrator
def get_target_versions_for_dataset(args, dataset_name: str) -> Optional[List[int]]:
    """Resolves dataset-specific target versions based on CLI flags or top-level defaults."""
    ds_lower = dataset_name.lower()
    if "pubmedqa" in ds_lower and args.pubmedqa_versions is not None:
        return args.pubmedqa_versions
    if "bioasq" in ds_lower and args.bioasq_versions is not None:
        return args.bioasq_versions
    return args.versions

def main():
    parser = argparse.ArgumentParser(description="Multi-Seed MedFact-EWC (Trust-Weighted Regularization) Continual Learning Pipeline")
    parser.add_argument("--datasets", nargs="+", default=["PubMedQA", "BioASQ2025"], choices=["BioASQ2025", "PubMedQA"])
    parser.add_argument("--techniques", nargs="+", default=TECHNIQUES)
    parser.add_argument("--mode", default="all", choices=["clean", "poisoned", "all"])
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the BioASQ2025/ and PubMedQA/ folders")
    
    # Version filtering: universal or per-dataset
    parser.add_argument("--versions", nargs="+", type=int, default=DEFAULT_VERSIONS, help="Universal version filter applied across datasets")
    parser.add_argument("--pubmedqa_versions", nargs="+", type=int, default=DEFAULT_PUBMEDQA_VERSIONS, help="Specific version numbers for PubMedQA (e.g. 1 2 3)")
    parser.add_argument("--bioasq_versions", nargs="+", type=int, default=DEFAULT_BIOASQ_VERSIONS, help="Specific version numbers for BioASQ2025 (e.g. 5 6 7)")

    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_root", type=str, default="results/MedFact_EWC")
    parser.add_argument("--aggregate_only", action="store_true")

    # Fact-checked clean stream (revision)
    parser.add_argument("--clean_trust_dir", type=str, default=None,
                        help="Directory with fact-checked CLEAN splits (s1.json, ...). Default: "
                             "<data_dir>/<dataset>/FactCheckedData/Clean_Fact_Checked. '{dataset}' is replaced by the dataset name.")
    parser.add_argument("--clean_trust_extra", nargs="+", default=None,
                        help="Optional JSON files with additional fact-checked clean rows (question/context/answer/trust_score). "
                             "Use '{dataset}' as placeholder.")
    parser.add_argument("--check_clean_trust", action="store_true",
                        help="Only report trust coverage of the clean stream and export samples still to be fact-checked.")
    parser.add_argument("--diagnose_clean_trust", action="store_true",
                        help="Only explain why clean rows do not match their fact-checked copies (no training).")
    parser.add_argument("--clean_subdir", type=str, default=CLEAN_FACTCHECKED_SUBDIR,
                        help="Output sub-directory for the fact-checked clean run (used by aggregation).")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=OPTIMAL_HPARAMS["num_train_epochs"])
    parser.add_argument("--lr", type=float, default=OPTIMAL_HPARAMS["learning_rate"])
    parser.add_argument("--batch_size", type=int, default=OPTIMAL_HPARAMS["per_device_train_batch_size"])
    parser.add_argument("--weight_decay", type=float, default=OPTIMAL_HPARAMS["weight_decay"])
    parser.add_argument("--warmup_ratio", type=float, default=OPTIMAL_HPARAMS["warmup_ratio"])
    parser.add_argument("--patience", type=int, default=OPTIMAL_HPARAMS["patience"])
    parser.add_argument("--max_src_len", type=int, default=OPTIMAL_HPARAMS["max_src_len"])
    parser.add_argument("--semantic_margin", type=float, default=OPTIMAL_HPARAMS["semantic_margin"])
    parser.add_argument("--margin_lambda", type=float, default=OPTIMAL_HPARAMS["margin_lambda"])

    # EWC Specific Hyperparameters
    parser.add_argument("--ewc_lambda", type=float, default=OPTIMAL_HPARAMS["ewc_lambda"], help="EWC quadratic loss penalty strength")
    parser.add_argument("--fisher_gamma", type=float, default=OPTIMAL_HPARAMS["fisher_gamma"], help="Online EWC Fisher moving average decay factor")
    parser.add_argument("--fisher_max_samples", type=int, default=OPTIMAL_HPARAMS["fisher_max_samples"], help="Sample limit for empirical Fisher calculation")
    parser.add_argument("--fisher_normalize", action="store_true", help="Normalize Fisher Information by its mean trace")
    parser.add_argument("--hparams_file", type=str, default=None)

    # Supervisor / Worker
    parser.add_argument("--no_supervisor", action="store_true")
    parser.add_argument("--max_restarts", type=int, default=50)
    parser.add_argument("--keep_snapshots", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()
    role = "worker" if args._worker else ("supervisor" if not (args.no_supervisor or args.aggregate_only or args.check_clean_trust or args.diagnose_clean_trust) else "main")
    print(f"[MedFact-EWC] started ({role}) | script: {os.path.abspath(__file__)} | cwd: {os.getcwd()}", flush=True)

    if not args._worker and not args.no_supervisor and not args.aggregate_only and not args.check_clean_trust and not args.diagnose_clean_trust:
        sys.exit(run_supervisor(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    live_csv = os.path.join(args.output_root, "live_split_metrics.csv")

    active_hparams = {
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "patience": args.patience,
        "max_src_len": args.max_src_len,
        "semantic_margin": args.semantic_margin,
        "margin_lambda": args.margin_lambda,
        "ewc_lambda": args.ewc_lambda,
        "fisher_gamma": args.fisher_gamma,
        "fisher_max_samples": args.fisher_max_samples,
        "fisher_normalize": args.fisher_normalize
    }

    if args.hparams_file and os.path.exists(args.hparams_file):
        active_hparams.update(load_json(args.hparams_file))
        print(f"[+] Loaded custom hyperparameters from {args.hparams_file}")

    print("\n" + "="*80)
    print(" MULTI-SEED MedFact-EWC (TRUST-WEIGHTED REGULARIZATION) PIPELINE")
    print(f" Model           : {args.model_name}")
    print(f" Device          : {device}")
    print(f" Datasets        : {args.datasets}")
    print(f" Techniques      : {args.techniques}")
    print(f" Seeds           : {args.seeds}")
    print(f" Output Root     : {args.output_root}")
    print(f" Margin / Lmb    : {active_hparams['semantic_margin']} / {active_hparams['margin_lambda']}")
    print(f" EWC Lambda (λ)  : {active_hparams['ewc_lambda']}")
    print(f" Fisher Gamma (γ): {active_hparams['fisher_gamma']}")
    print(f" Fisher Samples  : {active_hparams['fisher_max_samples']}")
    print("="*80 + "\n")

    def _fmt(path: Optional[str], ds: str) -> Optional[str]:
        return path.replace("{dataset}", ds) if path else path

    def _build_index(ds: str, ds_base: str) -> Dict[str, float]:
        extra = [_fmt(f, ds) for f in (args.clean_trust_extra or []) if os.path.exists(_fmt(f, ds))]
        index, info = build_clean_trust_index(
            ds_base, is_pubmedqa=(ds.lower() == "pubmedqa"),
            clean_trust_dir=_fmt(args.clean_trust_dir, ds), extra_files=extra
        )
        print(f"[*] Clean trust index for {ds}: {info['n_indexed']} samples indexed "
              f"({info['n_rows_with_trust']}/{info['n_rows_scanned']} fact-checked rows carry a trust score)")
        return index

    if args.aggregate_only:
        for ds in args.datasets:
            compute_multiseed_aggregations(args.output_root, ds, args.seeds, clean_subdir=args.clean_subdir)
        return

    if args.diagnose_clean_trust:
        for ds in args.datasets:
            diagnose_clean_trust(ds, resolve_dataset_dir(args.data_dir, ds))
        return

    def _clean_fc_dir(ds: str, ds_base: str) -> Optional[str]:
        d = resolve_clean_fc_dir(ds_base, _fmt(args.clean_trust_dir, ds))
        print(f"[*] Clean trust source for {ds}: " + (d if d else
              f"no {CLEAN_FC_DIRNAME} folder found -> matching clean samples against FactCheckedData"))
        return d

    if args.check_clean_trust:
        ok_all = True
        for ds in args.datasets:
            ds_base = resolve_dataset_dir(args.data_dir, ds)
            cfc = _clean_fc_dir(ds, ds_base)
            ok_all &= check_clean_trust_coverage(ds, ds_base, {} if cfc else _build_index(ds, ds_base),
                                                 report_dir=os.path.join(args.output_root, "_clean_trust_reports"),
                                                 clean_fc_dir=cfc)
        sys.exit(0 if ok_all else 2)

    chosen_techs = TECHNIQUES if "all" in args.techniques else args.techniques

    for ds in args.datasets:
        ds_base = resolve_dataset_dir(args.data_dir, ds)
        ds_out = os.path.join(args.output_root, ds)
        ds_versions = get_target_versions_for_dataset(args, ds)

        print(f"[*] Dataset: {ds} | Resolved Directory: {ds_base}")
        if ds_versions is not None:
            print(f"[*] Target Versions for {ds}: {ds_versions}")
        clean_fc_dir = _clean_fc_dir(ds, ds_base) if args.mode in ["clean", "all"] else None
        clean_trust_index = (_build_index(ds, ds_base) if (args.mode in ["clean", "all"] and not clean_fc_dir) else {})

        for s_idx, current_seed in enumerate(args.seeds, start=1):
            seed_all(current_seed)
            print(f"\n{'#'*80}\n [Seed {current_seed} ({s_idx}/{len(args.seeds)})] Executing MedFact-EWC Conditions for {ds}\n{'#'*80}")

            if args.mode in ["clean", "all"]:
                run_clean_baseline_seed(
                    dataset_name=ds, seed=current_seed, base_dir=ds_base,
                    out_dir_seed=os.path.join(ds_out, args.clean_subdir, f"seed_{current_seed}"),
                    live_csv=live_csv, model_name=args.model_name, device=device, hparams=active_hparams,
                    trust_index=clean_trust_index, clean_fc_dir=clean_fc_dir
                )

            if args.mode in ["poisoned", "all"]:
                for tech in chosen_techs:
                    run_poisoned_versions_seed(
                        dataset_name=ds, technique=tech, seed=current_seed, base_dir=ds_base,
                        out_dir_tech_seed=os.path.join(ds_out, tech, f"seed_{current_seed}"),
                        live_csv=live_csv, model_name=args.model_name, device=device, hparams=active_hparams,
                        versions=ds_versions, keep_snapshots=args.keep_snapshots
                    )

        compute_multiseed_aggregations(args.output_root, ds, args.seeds, clean_subdir=args.clean_subdir)

    print("\n" + "="*80)
    print(" ALL MedFact-EWC EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()