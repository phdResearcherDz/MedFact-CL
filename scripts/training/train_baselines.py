#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comparative CL baselines: isolated fine-tuning, O-LoRA, BiLoRA, iCaRL, spectral signatures.
Usage:
    python scripts/train_baselines.py --data_dir data --output_root results
Run with --help for all options.
"""

import os
import sys
import re
import gc
import csv
import json
import copy
import math
import time
import random
import argparse
import datetime
import subprocess
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
    set_seed,
)
from transformers.utils import logging as hf_logging

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

hf_logging.set_verbosity_error()

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# Defaults & Hyperparameters
DEFAULT_MODEL_NAME = "google/flan-t5-xl"

ALL_METHODS = ["isolated_ft", "o_lora", "bilora", "icarl", "spectral_signatures"]
METHOD_OUT_DIRS = {
    "o_lora": "results/OLoRA",
    "bilora": "results/BiLoRA",
    "isolated_ft": "results/Isolated",
    "icarl": "results/iCaRL",
    "spectral_signatures": "results/SpectralSignatures",
}
DEFAULT_OUT_TAG = ""

DEFAULT_SEEDS = [42, 123, 456, 789, 1011]
TECHNIQUES = ["negation_injection", "factual_reversal", "claim_deletion", "hallucinated_claim_injection"]
DEFAULT_DATASETS = ["BioASQ2025", "PubMedQA"]
DEFAULT_BIOASQ_VERSIONS = None
DEFAULT_PUBMEDQA_VERSIONS = None

MAX_SRC_LEN_BY_DATASET = {"bioasq2025": 1024, "pubmedqa": 512}

OPTIMAL_HPARAMS = {
    "learning_rate_ft": 1.0e-4,
    "learning_rate_peft": 5.0e-4,
    "num_train_epochs": 15,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "patience": 3,
    "min_epochs": 5,
    "max_src_len": 512,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q", "v"],
    "orth_lambda_olora": 0.5,
    "bilora_slot_rank": 4,
    "icarl_buffer_capacity": 80,
    "icarl_distill_lambda": 0.5,
    "icarl_distill_temp": 2.0,
    "spectral_poison_budget": 0.15,
    "spectral_min_class": 30,
    "spectral_warmup_epochs": 2,
}


# Helpers & Utilities
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + f".tmp_{os.getpid()}_{random.randint(1000, 9999)}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
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


_WARNED_LABELS = set()


def normalize_label(ans: Any, is_pubmedqa: bool = False) -> str:
    ans_str = str(ans or "").strip().lower()
    if ans_str.startswith("y"):
        return "yes"
    if ans_str.startswith("n"):
        return "no"
    if is_pubmedqa and (ans_str.startswith("m") or "maybe" in ans_str or "uncertain" in ans_str):
        return "maybe"
    if ans_str not in _WARNED_LABELS:
        _WARNED_LABELS.add(ans_str)
        print(f"  [WARN] Unrecognised label {ans_str!r} mapped to 'no'. Check your data.", flush=True)
    return "no"


def get_candidate_info(tokenizer: AutoTokenizer, is_pubmedqa: bool) -> Tuple[List[str], List[int]]:
    cand_names = ["no", "yes", "maybe"] if is_pubmedqa else ["no", "yes"]
    cand_ids = [tokenizer.encode(name, add_special_tokens=False)[0] for name in cand_names]
    return cand_names, cand_ids


def get_decoder_start_id(model: nn.Module, tokenizer: AutoTokenizer) -> int:
    start = getattr(model.config, "decoder_start_token_id", None)
    return int(start) if start is not None else int(tokenizer.pad_token_id)


def resolve_dataset_dir(base_workspace: str, dataset_name: str) -> str:
    path = os.path.join(base_workspace, dataset_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset folder not found: {path}")
    return path


def sorted_split_files(directory: str) -> List[str]:
    return sorted(
        [f for f in os.listdir(directory) if f.startswith("s") and f.endswith(".json")],
        key=lambda x: int(re.search(r"\d+", x).group()),
    )


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# Crash-Safe Rolling Checkpoints & State Resolution
def save_rolling_checkpoint(model: nn.Module, tokenizer: AutoTokenizer, ckpt_dir: str, completed_split: int):
    """Saves model weights/adapters and tokenizer atomically with a progress marker."""
    os.makedirs(ckpt_dir, exist_ok=True)
    marker_file = os.path.join(ckpt_dir, "progress.json")
    if os.path.exists(marker_file):
        try:
            os.remove(marker_file)
        except Exception:
            pass

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(ckpt_dir)
    else:
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(ckpt_dir)

    save_json({
        "completed_split": int(completed_split),
        "timestamp": datetime.datetime.now().isoformat()
    }, marker_file)


def resolve_rolling_checkpoint(ckpt_dir: str) -> Tuple[bool, int]:
    """Inspects checkpoint directory and returns (exists, completed_split)."""
    if not os.path.isdir(ckpt_dir):
        return False, 0
    marker = os.path.join(ckpt_dir, "progress.json")
    if os.path.exists(marker):
        try:
            data = load_json(marker)
            return True, int(data["completed_split"])
        except Exception:
            pass
    return False, 0


# Prompt Formatting & Telemetry Box
def format_prompt(
    question: str,
    context: str,
    is_pubmedqa: bool = False,
    tokenizer: Optional[AutoTokenizer] = None,
    max_src_len: int = 512,
) -> str:
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


def truncation_rate(rows: List[Dict[str, Any]], tokenizer: AutoTokenizer, is_pubmedqa: bool, max_src_len: int) -> float:
    n = 0
    for r in rows:
        full = format_prompt(r.get("question"), r.get("context"), is_pubmedqa=is_pubmedqa, tokenizer=None)
        if len(tokenizer.encode(full, add_special_tokens=False)) + 1 > max_src_len:
            n += 1
    return n / max(len(rows), 1)


def print_metric_box(header_text: str, metrics: Dict[str, Any], duration_sec: float = 0.0, method_name: str = ""):
    acc = metrics.get("accuracy", 0.0)
    f1 = metrics.get("f1_macro", 0.0)
    prec = metrics.get("precision_macro", 0.0)
    rec = metrics.get("recall_macro", 0.0)
    auroc = metrics.get("auroc")
    brier = metrics.get("brier_score")
    dist = metrics.get("pred_distribution", {})
    acc_ci = metrics.get("accuracy_ci95")
    f1_ci = metrics.get("f1_macro_ci95")

    auroc_str = f"{auroc:.4f}" if isinstance(auroc, (int, float)) else "N/A"
    brier_str = f"{brier:.4f}" if isinstance(brier, (int, float)) else "N/A"
    dist_str = " | ".join([f"{k}: {v}" for k, v in dist.items()])

    line_w = 80
    print("\n" + "┌" + "─" * (line_w - 2) + "┐")
    print(f"│  {header_text:<{line_w - 5}} │")
    print("├" + "─" * (line_w - 2) + "┤")
    print(f"│  Accuracy: {acc:.4f} ({acc*100:5.2f}%)   |  Macro F1: {f1:.4f} ({f1*100:5.2f}%)                   │")
    print(f"│  Precision: {prec:.4f}          |  Recall:   {rec:.4f}                             │")
    print(f"│  AUROC:     {auroc_str:<14}  |  Brier:    {brier_str:<14}  [{method_name} | {duration_sec:.1f}s]   │")
    if acc_ci and f1_ci:
        ci_line = f"Acc 95% CI [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]  F1 95% CI [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]"
        print(f"│  {ci_line:<{line_w - 5}} │")
    if dist_str:
        print(f"│  Preds:     {dist_str:<64} │")
    print("└" + "─" * (line_w - 2) + "┘\n", flush=True)


LIVE_CSV_HEADER = [
    "timestamp", "seed", "dataset", "setting", "technique", "version",
    "split", "total_splits", "accuracy", "f1_macro", "precision_macro",
    "recall_macro", "auroc", "brier_score", "method", "duration_sec",
]
_CSV_KEY_COLS = ["seed", "dataset", "setting", "technique", "version", "split", "method"]


def append_live_split_csv(csv_path: str, row: Dict[str, Any]):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    clean_row = {k: ("" if row.get(k) is None else row.get(k)) for k in LIVE_CSV_HEADER}
    key = tuple(str(clean_row[k]) for k in _CSV_KEY_COLS)

    kept: List[Dict[str, Any]] = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                for old in csv.DictReader(f):
                    if tuple(str(old.get(k, "")) for k in _CSV_KEY_COLS) == key:
                        continue
                    kept.append({k: old.get(k, "") for k in LIVE_CSV_HEADER})
        except Exception:
            kept = []

    tmp_path = csv_path + f".tmp_{os.getpid()}_{random.randint(1000, 9999)}"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LIVE_CSV_HEADER)
        w.writeheader()
        w.writerows(kept)
        w.writerow(clean_row)
    os.replace(tmp_path, csv_path)


# Dataset Definition (Lazy Formatting)
class Seq2SeqQADataset(TorchDataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        is_pubmedqa: bool = False,
        max_src_len: int = 512,
        max_tgt_len: int = 8,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.is_pubmedqa = is_pubmedqa
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        cand_names, cand_ids = get_candidate_info(tokenizer, is_pubmedqa)
        self.label2token_id = {name: cid for name, cid in zip(cand_names, cand_ids)}
        self.default_cid = cand_ids[0]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        q = r.get("question", "")
        c = r.get("context", "")
        ans = normalize_label(r.get("answer"), is_pubmedqa=self.is_pubmedqa)
        prompt = format_prompt(
            q, c, is_pubmedqa=self.is_pubmedqa, tokenizer=self.tokenizer, max_src_len=self.max_src_len
        )
        target_token_id = self.label2token_id.get(ans, self.default_cid)
        return {
            "prompt": prompt,
            "target_text": ans,
            "target_token_id": target_token_id,
            "id": str(r.get("id", "")),
        }


def make_collate_fn(tokenizer: AutoTokenizer, max_src_len: int = 512, max_tgt_len: int = 8):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        targets = [item["target_text"] for item in batch]

        model_inputs = tokenizer(
            prompts, padding=True, truncation=True, max_length=max_src_len, return_tensors="pt"
        )
        labels = tokenizer(
            text_target=targets, padding=True, truncation=True, max_length=max_tgt_len, return_tensors="pt"
        )
        labels_ids = labels["input_ids"].clone()
        labels_ids[labels_ids == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels_ids
        model_inputs["target_token_id"] = torch.tensor([item["target_token_id"] for item in batch], dtype=torch.long)
        return model_inputs
    return collate_fn


# Model Evaluation
def _macro_f1_np(g: np.ndarray, p: np.ndarray) -> float:
    f = []
    for c in np.union1d(g, p):
        tp = np.sum((g == c) & (p == c))
        fp = np.sum((g != c) & (p == c))
        fn = np.sum((g == c) & (p != c))
        d = 2 * tp + fp + fn
        f.append(2 * tp / d if d > 0 else 0.0)
    return float(np.mean(f)) if f else 0.0


def bootstrap_ci(gold: List[int], pred: List[int], n_boot: int = 1000, seed: int = 0):
    rng = np.random.RandomState(seed)
    g, p = np.asarray(gold), np.asarray(pred)
    n = len(g)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        accs.append(float(np.mean(g[idx] == p[idx])))
        f1s.append(_macro_f1_np(g[idx], p[idx]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return ci(accs), ci(f1s)


def evaluate_model(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    test_rows: List[Dict[str, Any]],
    is_pubmedqa: bool,
    device: torch.device,
    batch_size: int = 8,
    max_src_len: int = 512,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    cand_names, cand_token_ids = get_candidate_info(tokenizer, is_pubmedqa)
    label2id = {l: i for i, l in enumerate(cand_names)}
    cand_t = torch.tensor(cand_token_ids, dtype=torch.long, device=device)

    gold_labels = [normalize_label(r.get("answer"), is_pubmedqa=is_pubmedqa) for r in test_rows]
    gold_ids = [label2id[g] for g in gold_labels]

    pred_ids: List[int] = []
    yes_probs: List[float] = []
    all_probs_list: List[List[float]] = []
    predictions_log: List[Dict[str, Any]] = []

    dec_start_id = get_decoder_start_id(model, tokenizer)

    with torch.inference_mode():
        for i in range(0, len(test_rows), batch_size):
            b_rows = test_rows[i:i + batch_size]
            b_prompts = [
                format_prompt(r.get("question"), r.get("context"), is_pubmedqa=is_pubmedqa,
                              tokenizer=tokenizer, max_src_len=max_src_len)
                for r in b_rows
            ]
            inputs = tokenizer(
                b_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_src_len
            ).to(device)

            bs = len(b_prompts)
            dec_in = torch.full((bs, 1), dec_start_id, dtype=torch.long, device=device)
            outputs = model(**inputs, decoder_input_ids=dec_in, use_cache=False)

            next_logits = outputs.logits[:, 0, :]
            cand_logits = next_logits.index_select(1, cand_t).float()
            cand_logits = torch.nan_to_num(cand_logits, nan=0.0, posinf=30.0, neginf=-30.0)
            cand_probs = F.softmax(cand_logits, dim=-1)
            cand_probs = torch.nan_to_num(cand_probs, nan=1.0 / len(cand_token_ids))

            p_np = cand_probs.cpu().numpy()
            l_np = cand_logits.cpu().numpy()

            del outputs, inputs, dec_in, next_logits, cand_logits, cand_probs

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

    pred_counts = Counter(pred_ids)
    pred_dist = {cand_names[idx]: pred_counts.get(idx, 0) for idx in range(len(cand_names))}

    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(gold_ids, pred_ids)), 4),
        "f1_macro": round(float(f1_score(gold_ids, pred_ids, average="macro", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(gold_ids, pred_ids, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(gold_ids, pred_ids, average="macro", zero_division=0)), 4),
        "pred_distribution": pred_dist,
    }
    acc_ci, f1_ci = bootstrap_ci(gold_ids, pred_ids)
    metrics["accuracy_ci95"] = acc_ci
    metrics["f1_macro_ci95"] = f1_ci

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
    except Exception:
        metrics["auroc"] = None
        metrics["brier_score"] = None

    return metrics, predictions_log


def compute_dev_loss(model: nn.Module, dev_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.inference_mode():
        for batch in dev_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
            outputs = model(**inputs, use_cache=False)
            loss = outputs.loss
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                steps += 1
            del outputs, loss, inputs
    return total_loss / max(steps, 1)


# LoRA & Orthogonality Utilities
def create_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05,
                       target_modules: Optional[List[str]] = None) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or ["q", "v"],
    )


def strip_adapter_name(name: str) -> str:
    return re.sub(r"(lora_[AB])\.[^.]+\.weight", r"\1.weight", name)


def set_active_adapters(model: nn.Module, names: List[str], strict: bool = False) -> bool:
    arg = names if len(names) > 1 else names[0]
    last_err: Optional[Exception] = None
    for target in (getattr(model, "base_model", None), model):
        if target is None or not hasattr(target, "set_adapter"):
            continue
        try:
            target.set_adapter(arg)
            act = getattr(model, "active_adapters", None)
            if act is not None:
                act_list = [act] if isinstance(act, str) else list(act)
                if set(act_list) != set(names):
                    print(f"  [WARN] Requested adapters {names} but model reports {act_list}.", flush=True)
            return True
        except Exception as e:
            last_err = e
    msg = f"Could not activate adapters {names}: {last_err}"
    if strict:
        raise RuntimeError(msg + " (multi-adapter set_adapter unsupported by this PEFT version; upgrade peft)")
    print("  [WARN] " + msg, flush=True)
    return False


def freeze_except(model: nn.Module, trainable_fn):
    for n, p in model.named_parameters():
        p.requires_grad = bool(trainable_fn(n))


def compute_olora_orth_loss(
    curr_A: Dict[str, torch.Tensor],
    past_A_list: List[Dict[str, torch.Tensor]],
    orth_lambda: float,
) -> torch.Tensor:
    total = None
    for past in past_A_list:
        for key, a_curr in curr_A.items():
            a_past = past[key].to(dtype=a_curr.dtype)
            term = torch.mm(a_past, a_curr.t()).float().pow(2).sum()
            total = term if total is None else total + term
    if total is None:
        return torch.tensor(0.0)
    return orth_lambda * total


def apply_bilora_masks(model: nn.Module, split_idx: int, total_splits: int):
    clear_bilora_hooks(model)
    model._bilora_hooks = []
    for name, param in model.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        is_A = "lora_A" in name
        r = param.shape[0] if is_A else param.shape[1]
        slot = max(1, r // total_splits)
        start = (split_idx - 1) * slot
        end = r if split_idx == total_splits else min(r, start + slot)

        mask = torch.zeros_like(param)
        if is_A:
            mask[start:end, :] = 1.0
        else:
            mask[:, start:end] = 1.0
        model._bilora_hooks.append(param.register_hook(lambda g, m=mask: g * m))


def clear_bilora_hooks(model: nn.Module):
    if hasattr(model, "_bilora_hooks"):
        for h in model._bilora_hooks:
            h.remove()
    model._bilora_hooks = []


# iCaRL & Spectral Signatures
class ICaRLReplayBuffer:
    def __init__(self, capacity: int = 80, seed: int = 42):
        self.capacity = capacity
        self.seed = seed
        self.exemplars_by_class: Dict[str, List[Dict[str, Any]]] = {}

    @staticmethod
    def herding_selection(samples: List[Dict[str, Any]], embeddings: np.ndarray, m_per_class: int) -> List[Dict[str, Any]]:
        if len(samples) <= m_per_class:
            return list(samples)

        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        mu = np.mean(embeddings, axis=0)
        selected: List[int] = []
        current_sum = np.zeros_like(mu)

        for i in range(1, m_per_class + 1):
            cand_mean = (current_sum + embeddings) / i
            distances = np.linalg.norm(cand_mean - mu, axis=1)
            for idx in selected:
                distances[idx] = float("inf")
            best = int(np.argmin(distances))
            selected.append(best)
            current_sum += embeddings[best]

        return [samples[idx] for idx in selected]

    def update(self, new_rows, model, tokenizer, device, is_pubmedqa: bool, max_src_len: int = 512):
        model.eval()
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in new_rows:
            groups.setdefault(normalize_label(r.get("answer"), is_pubmedqa=is_pubmedqa), []).append(r)

        all_classes = sorted(set(self.exemplars_by_class) | set(groups))
        if not all_classes:
            return
        m_per_class = max(1, self.capacity // len(all_classes))
        enc = getattr(model, "encoder", None) or model.get_encoder()

        for c in all_classes:
            pool = list(self.exemplars_by_class.get(c, [])) + list(groups.get(c, []))
            if not pool:
                continue
            embs = []
            with torch.inference_mode():
                for b in range(0, len(pool), 8):
                    b_rows = pool[b:b + 8]
                    prompts = [
                        format_prompt(r.get("question"), r.get("context"), is_pubmedqa=is_pubmedqa,
                                      tokenizer=tokenizer, max_src_len=max_src_len)
                        for r in b_rows
                    ]
                    inps = tokenizer(prompts, return_tensors="pt", padding=True,
                                     truncation=True, max_length=max_src_len).to(device)
                    out = enc(input_ids=inps["input_ids"], attention_mask=inps["attention_mask"])
                    hidden = out.last_hidden_state.float()
                    mask = inps["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                    mean_e = (torch.sum(hidden * mask, 1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy()
                    embs.append(mean_e)
                    del inps, out, hidden, mask
            embs = np.concatenate(embs, axis=0)
            self.exemplars_by_class[c] = self.herding_selection(pool, embs, m_per_class)

    def get_all(self) -> List[Dict[str, Any]]:
        combined: List[Dict[str, Any]] = []
        for c_samples in self.exemplars_by_class.values():
            combined.extend(c_samples)
        random.Random(self.seed).shuffle(combined)
        return combined


class SpectralSanitizer:
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer, device: torch.device,
                 poison_budget: float = 0.15, min_class: int = 30, max_src_len: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.poison_budget = poison_budget
        self.min_class = min_class
        self.max_src_len = max_src_len

    def _embed(self, rows: List[Dict[str, Any]], is_pubmedqa: bool) -> np.ndarray:
        self.model.eval()
        dec_start_id = get_decoder_start_id(self.model, self.tokenizer)
        out_embs = []

        last_hidden: Dict[str, torch.Tensor] = {}
        def _hook(module, inp, out):
            last_hidden["val"] = out.detach()

        dec_norm = self.model.decoder.final_layer_norm
        handle = dec_norm.register_forward_hook(_hook)

        try:
            with torch.inference_mode():
                for i in range(0, len(rows), 4):
                    b_rows = rows[i:i + 4]
                    prompts = [
                        format_prompt(r.get("question"), r.get("context"), is_pubmedqa=is_pubmedqa,
                                      tokenizer=self.tokenizer, max_src_len=self.max_src_len)
                        for r in b_rows
                    ]
                    inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                            truncation=True, max_length=self.max_src_len).to(self.device)
                    bs = inputs["input_ids"].size(0)
                    dec_in = torch.full((bs, 1), dec_start_id, dtype=torch.long, device=self.device)
                    _ = self.model(**inputs, decoder_input_ids=dec_in, use_cache=False)
                    out_embs.append(last_hidden["val"][:, 0, :].float().cpu().numpy())
                    del inputs, dec_in
                    last_hidden.clear()
        finally:
            handle.remove()

        return np.concatenate(out_embs, axis=0)

    def filter_data(self, rows: List[Dict[str, Any]], is_pubmedqa: bool = False) -> List[Dict[str, Any]]:
        if len(rows) < 20:
            return rows

        embs = self._embed(rows, is_pubmedqa)
        labels = [normalize_label(r.get("answer"), is_pubmedqa=is_pubmedqa) for r in rows]
        bad: set = set()
        skipped: List[str] = []

        for lbl in sorted(set(labels)):
            cls_idx = np.array([i for i, l in enumerate(labels) if l == lbl])
            if len(cls_idx) < self.min_class:
                skipped.append(f"{lbl}({len(cls_idx)})")
                continue
            M = embs[cls_idx] - embs[cls_idx].mean(axis=0, keepdims=True)
            try:
                _, _, vh = np.linalg.svd(M, full_matrices=False)
                scores = np.square(M @ vh[0])
                n_remove = int(round(1.5 * self.poison_budget * len(cls_idx)))
                if n_remove > 0:
                    top = np.argsort(-scores)[:n_remove]
                    bad.update(int(cls_idx[t]) for t in top)
            except Exception as e:
                print(f"  [Spectral Signatures] Class '{lbl}' SVD exception: {e}")

        kept = [r for i, r in enumerate(rows) if i not in bad]
        msg = f"  [Spectral Signatures] Removed {len(bad)}/{len(rows)} samples."
        if skipped:
            msg += f" Skipped small classes: {', '.join(skipped)}."
        print(msg, flush=True)
        return kept


# Split Training Engine
def train_single_split(
    method: str,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    train_rows: List[Dict[str, Any]],
    dev_rows: List[Dict[str, Any]],
    is_pubmedqa: bool,
    hparams: Dict[str, Any],
    seed: int,
    device: torch.device,
    split_idx: int = 1,
    total_splits: int = 1,
    adapter_name: Optional[str] = None,
    past_adapter_states: Optional[List[Dict[str, torch.Tensor]]] = None,
    teacher_model: Optional[nn.Module] = None,
) -> nn.Module:
    seed_all(seed)

    is_lora = method in ("o_lora", "bilora")
    lr = float(hparams["learning_rate_peft"] if is_lora else hparams["learning_rate_ft"])
    epochs = int(hparams["num_train_epochs"])
    min_epochs = int(hparams.get("min_epochs", 5))
    batch_size = int(hparams["per_device_train_batch_size"])
    accum_steps = int(hparams["gradient_accumulation_steps"])
    max_src_len = int(hparams["max_src_len"])
    patience = int(hparams["patience"])
    weight_decay = 0.0 if method == "bilora" else float(hparams["weight_decay"])

    train_ds = Seq2SeqQADataset(train_rows, tokenizer, is_pubmedqa=is_pubmedqa, max_src_len=max_src_len)
    dev_ds = Seq2SeqQADataset(dev_rows, tokenizer, is_pubmedqa=is_pubmedqa, max_src_len=max_src_len)
    collate_fn = make_collate_fn(tokenizer, max_src_len=max_src_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn)

    if method == "bilora":
        apply_bilora_masks(model, split_idx=split_idx, total_splits=total_splits)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"FATAL: No trainable parameters found for {method} at split {split_idx}!")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * float(hparams["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    orth_lambda = float(hparams["orth_lambda_olora"])
    curr_A: Dict[str, torch.Tensor] = {}
    past_A_dev: List[Dict[str, torch.Tensor]] = []
    if method == "o_lora" and past_adapter_states:
        curr_A = {strip_adapter_name(n): p for n, p in model.named_parameters()
                  if p.requires_grad and "lora_A" in n}
        past_A_dev = [{k: v.to(device) for k, v in sd.items()} for sd in past_adapter_states]

    is_icarl = (method == "icarl" and teacher_model is not None)
    distill_lambda = float(hparams["icarl_distill_lambda"])
    distill_temp = float(hparams["icarl_distill_temp"])
    _, cand_token_ids = get_candidate_info(tokenizer, is_pubmedqa)
    cand_t = torch.tensor(cand_token_ids, dtype=torch.long, device=device)

    resolved_adapter = adapter_name or getattr(model, "active_adapter", "default")

    def snapshot():
        if is_lora:
            sd = get_peft_model_state_dict(model, adapter_name=resolved_adapter)
            return {k: v.detach().cpu().clone() for k, v in sd.items()}
        return {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}

    best_loss = float("inf")
    patience_cnt = 0
    best_state = None

    epoch_pbar = tqdm(range(1, epochs + 1), desc=f"  [Split {split_idx} Training]", leave=False)
    for epoch in epoch_pbar:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0

        for b_idx, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
            outputs = model(**inputs, use_cache=False)
            loss = outputs.loss

            if is_icarl:
                with torch.no_grad():
                    teacher_outputs = teacher_model(**inputs, use_cache=False)
                s_logits = outputs.logits[:, 0, :].index_select(1, cand_t).float()
                t_logits = teacher_outputs.logits[:, 0, :].index_select(1, cand_t).float()
                s_logp = F.log_softmax(s_logits / distill_temp, dim=-1)
                t_p = F.softmax(t_logits / distill_temp, dim=-1)
                loss_kd = F.kl_div(s_logp, t_p, reduction="batchmean") * (distill_temp ** 2)
                loss = (1.0 - distill_lambda) * loss + distill_lambda * loss_kd.to(loss.dtype)
                del teacher_outputs, s_logits, t_logits, s_logp, t_p, loss_kd

            if past_A_dev:
                orth_term = compute_olora_orth_loss(curr_A, past_A_dev, orth_lambda).to(loss.device, dtype=loss.dtype)
                loss = loss + orth_term

            (loss / accum_steps).backward()
            epoch_loss += loss.item()
            del outputs, loss, inputs

            if (b_idx + 1) % accum_steps == 0 or (b_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        dev_loss = compute_dev_loss(model, dev_loader, device)
        epoch_pbar.set_postfix({
            "train_loss": f"{avg_train_loss:.4f}",
            "dev_loss": f"{dev_loss:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

        if dev_loss < best_loss:
            best_loss = dev_loss
            patience_cnt = 0
            best_state = snapshot()
        elif epoch > min_epochs:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state is not None:
        if is_lora:
            set_peft_model_state_dict(model, best_state, adapter_name=resolved_adapter)
        else:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if method == "bilora":
        clear_bilora_hooks(model)

    del optimizer, scheduler, best_state, train_loader, dev_loader, train_ds, dev_ds, past_A_dev, curr_A, cand_t
    free_memory()
    return model


# Model Construction & Sanitization
def load_fresh_model(base_model_name: str, device: torch.device) -> nn.Module:
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return model.to(device)


def build_or_extend_model(
    method: str,
    current_model: Optional[nn.Module],
    base_model_name: str,
    hp: Dict[str, Any],
    seed: int,
    device: torch.device,
    s_idx: int,
    total_splits: int,
) -> Tuple[nn.Module, Optional[str]]:
    seed_all(seed * 1000 + s_idx)

    if method == "o_lora":
        ad = f"split_{s_idx}"
        cfg = create_lora_config(hp["lora_r"], hp["lora_alpha"], hp["lora_dropout"], hp["target_modules"])
        if current_model is None:
            current_model = get_peft_model(load_fresh_model(base_model_name, device), cfg, adapter_name=ad)
        else:
            current_model.add_adapter(ad, cfg)
        set_active_adapters(current_model, [f"split_{i}" for i in range(1, s_idx + 1)], strict=True)
        freeze_except(current_model, lambda n: "lora_" in n and f".{ad}." in n)
        return current_model, ad

    if method == "bilora":
        ad = "bilora"
        if current_model is None:
            r_total = int(hp["bilora_slot_rank"]) * total_splits
            cfg = create_lora_config(r_total, 2 * r_total, hp["lora_dropout"], hp["target_modules"])
            current_model = get_peft_model(load_fresh_model(base_model_name, device), cfg, adapter_name=ad)
        freeze_except(current_model, lambda n: "lora_" in n)
        return current_model, ad

    if method == "isolated_ft" or current_model is None:
        current_model = load_fresh_model(base_model_name, device)
    if int(hp.get("max_src_len", 512)) > 512 and hasattr(current_model, "gradient_checkpointing_enable"):
        current_model.gradient_checkpointing_enable()
    for p in current_model.parameters():
        p.requires_grad = True
    return current_model, None


def spectral_sanitize_split(
    current_model: Optional[nn.Module],
    base_model_name: str,
    tokenizer: AutoTokenizer,
    rows: List[Dict[str, Any]],
    dev_rows: List[Dict[str, Any]],
    is_pubmedqa: bool,
    hp: Dict[str, Any],
    seed: int,
    device: torch.device,
    s_idx: int,
) -> List[Dict[str, Any]]:
    if len(rows) < 20:
        return rows
    seed_all(seed * 1000 + s_idx)

    if current_model is not None:
        current_model.to("cpu")
        free_memory()
        warm = copy.deepcopy(current_model).to(device)
    else:
        warm = load_fresh_model(base_model_name, device)

    if int(hp.get("max_src_len", 512)) > 512 and hasattr(warm, "gradient_checkpointing_enable"):
        warm.gradient_checkpointing_enable()
    for p in warm.parameters():
        p.requires_grad = True

    hp_w = dict(hp)
    hp_w["num_train_epochs"] = int(hp["spectral_warmup_epochs"])
    hp_w["min_epochs"] = int(hp["spectral_warmup_epochs"])
    warm = train_single_split(
        method="spectral_signatures", model=warm, tokenizer=tokenizer, train_rows=rows, dev_rows=dev_rows,
        is_pubmedqa=is_pubmedqa, hparams=hp_w, seed=seed, device=device, split_idx=s_idx,
    )
    if hasattr(warm, "gradient_checkpointing_disable"):
        warm.gradient_checkpointing_disable()

    sanitizer = SpectralSanitizer(
        warm, tokenizer, device, poison_budget=hp["spectral_poison_budget"],
        min_class=int(hp["spectral_min_class"]), max_src_len=int(hp["max_src_len"]),
    )
    kept = sanitizer.filter_data(rows, is_pubmedqa=is_pubmedqa)
    del sanitizer, warm
    free_memory()

    if current_model is not None:
        current_model.to(device)

    return kept


# Timeline Runner with Auto-Resume Support
def run_timeline(
    *, method: str, src_dir: str, out_dir: str, setting: str, technique: str, version: str,
    n_poisoned: int, dataset_name: str, is_pubmedqa: bool,
    dev_rows: List[Dict[str, Any]], test_rows: List[Dict[str, Any]],
    hp: Dict[str, Any], seed: int, tokenizer: AutoTokenizer, base_model_name: str,
    device: torch.device, live_csv_path: str,
) -> Dict[str, Any]:
    split_files = sorted_split_files(src_dir)
    total = len(split_files)
    summary_file = os.path.join(out_dir, "all_splits_metrics.json")
    rolling_ckpt_dir = os.path.join(out_dir, "_last_checkpoint")
    tag = f"{method.upper()} | {dataset_name} | {technique} | {version} | Seed {seed}"

    if os.path.exists(summary_file) and all(
        is_valid_metrics_file(os.path.join(out_dir, f"s{i}", "metrics.json")) for i in range(1, total + 1)
    ):
        print(f"[✓ Skipped] {tag} is complete.")
        return load_json(summary_file)

    os.makedirs(out_dir, exist_ok=True)
    max_src_len = int(hp["max_src_len"])
    seed_all(seed)

    # Check for rolling checkpoints to resume mid-timeline
    has_ckpt, completed_split = resolve_rolling_checkpoint(rolling_ckpt_dir)
    start_split = 1
    all_metrics: Dict[str, Any] = {}
    past_A: List[Dict[str, torch.Tensor]] = []
    current_model: Optional[nn.Module] = None
    teacher_model: Optional[nn.Module] = None
    icarl_buf = ICaRLReplayBuffer(capacity=hp["icarl_buffer_capacity"], seed=seed)

    if has_ckpt and 1 <= completed_split < total:
        print(f"\n[⚡ RESUMING] {tag} found rolling checkpoint at split {completed_split}. Resuming from split {completed_split + 1}...", flush=True)
        start_split = completed_split + 1
        # Load completed split metrics
        for i in range(1, completed_split + 1):
            mf = os.path.join(out_dir, f"s{i}", "metrics.json")
            if is_valid_metrics_file(mf):
                all_metrics[f"s{i}"] = load_json(mf)
        
        # Load model state from rolling checkpoint
        if method in ("o_lora", "bilora"):
            current_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            ).to(device)
            # PEFT model loading logic
            current_model = PeftModel.from_pretrained(current_model, rolling_ckpt_dir, is_trainable=True)
        else:
            current_model = load_fresh_model(base_model_name, device)
            state_dict_path = os.path.join(rolling_ckpt_dir, "pytorch_model.bin")
            if os.path.exists(state_dict_path):
                current_model.load_state_dict(torch.load(state_dict_path, map_location=device))
    else:
        print(f"\n>>> [{tag}] Running {setting} timeline ({total} splits; starting from split 1) <<<", flush=True)

    for s_idx in range(start_split, total + 1):
        s_name = f"s{s_idx}"
        s_out = os.path.join(out_dir, s_name)
        os.makedirs(s_out, exist_ok=True)
        s_rows = load_json(os.path.join(src_dir, split_files[s_idx - 1]))
        train_data = list(s_rows)

        if method == "spectral_signatures":
            train_data = spectral_sanitize_split(current_model, base_model_name, tokenizer, train_data,
                                                 dev_rows, is_pubmedqa, hp, seed, device, s_idx)
        if method == "icarl" and s_idx > 1:
            train_data.extend(icarl_buf.get_all())

        current_model, active_ad = build_or_extend_model(method, current_model, base_model_name, hp,
                                                         seed, device, s_idx, total)

        t0 = time.time()
        current_model = train_single_split(
            method=method, model=current_model, tokenizer=tokenizer, train_rows=train_data,
            dev_rows=dev_rows, is_pubmedqa=is_pubmedqa, hparams=hp, seed=seed, device=device,
            split_idx=s_idx, total_splits=total, adapter_name=active_ad,
            past_adapter_states=past_A if method == "o_lora" else None,
            teacher_model=teacher_model if method == "icarl" else None,
        )
        duration = time.time() - t0

        metrics, preds = evaluate_model(current_model, tokenizer, test_rows, is_pubmedqa=is_pubmedqa,
                                        device=device, max_src_len=max_src_len)
        save_json(metrics, os.path.join(s_out, "metrics.json"))
        save_json(preds, os.path.join(s_out, "predictions.json"))
        all_metrics[s_name] = metrics
        del preds

        if setting == "clean":
            hdr = f"{dataset_name} | Clean Baseline | Split {s_idx}/{total} (Seed {seed})"
        else:
            split_tag = "POISONED" if s_idx <= n_poisoned else "CLEAN"
            hdr = f"{dataset_name} | {technique} {version} | Split {s_idx}/{total} ({split_tag} | Seed {seed})"
        print_metric_box(hdr, metrics, duration, method)

        append_live_split_csv(live_csv_path, {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": seed, "dataset": dataset_name, "setting": setting, "technique": technique,
            "version": version, "split": s_idx, "total_splits": total,
            "accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"],
            "precision_macro": metrics["precision_macro"], "recall_macro": metrics["recall_macro"],
            "auroc": metrics.get("auroc"), "brier_score": metrics.get("brier_score"),
            "method": method, "duration_sec": round(duration, 2),
        })

        # Save rolling checkpoint after each successful split completion
        save_rolling_checkpoint(current_model, tokenizer, rolling_ckpt_dir, s_idx)
        save_json(all_metrics, summary_file)

        # Continual-state updates
        if method == "o_lora":
            past_A.append({strip_adapter_name(n): p.detach().cpu().clone()
                           for n, p in current_model.named_parameters()
                           if "lora_A" in n and f".{active_ad}." in n})
        elif method == "isolated_ft":
            del current_model
            current_model = None
        elif method == "icarl":
            if teacher_model is not None:
                del teacher_model
                teacher_model = None
                free_memory()
            icarl_buf.update(s_rows, current_model, tokenizer, device,
                             is_pubmedqa=is_pubmedqa, max_src_len=max_src_len)
            if s_idx < total:
                cpu_sd = {k: v.detach().cpu().clone() for k, v in current_model.state_dict().items()}
                teacher_model = load_fresh_model(base_model_name, device)
                teacher_model.load_state_dict({k: v.to(device) for k, v in cpu_sd.items()})
                del cpu_sd
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False

        del s_rows, train_data
        free_memory()

    # Clean up rolling checkpoints once the full timeline completes successfully
    if os.path.exists(rolling_ckpt_dir):
        try:
            import shutil
            shutil.rmtree(rolling_ckpt_dir, ignore_errors=True)
        except Exception:
            pass

    print(f"[✓ Done] {tag} saved.")
    del current_model, teacher_model, past_A, icarl_buf
    free_memory()
    return all_metrics


# Benchmark Pipeline & Self-Healing Supervisor
def run_method_benchmark(
    method: str,
    datasets: List[str],
    techniques: List[str],
    bioasq_versions: List[int],
    pubmedqa_versions: List[int],
    seed: int,
    base_model_name: str,
    device: torch.device,
    out_tag: str = DEFAULT_OUT_TAG,
    data_dir: str = "data",
):
    out_root = METHOD_OUT_DIRS[method] + out_tag
    print("\n" + "=" * 80)
    print(f" [BENCHMARK RUN] Method: {method.upper()} | Seed: {seed}")
    print(f" Datasets: {datasets} | Techniques: {techniques}")
    print(f" Versions: BioASQ={bioasq_versions}, PubMedQA={pubmedqa_versions}")
    print(f" Root: {out_root}")
    print("=" * 80 + "\n", flush=True)

    os.makedirs(out_root, exist_ok=True)
    live_csv_path = os.path.join(out_root, "live_split_metrics.csv")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

    for dataset_name in datasets:
        is_pubmedqa = dataset_name.lower() == "pubmedqa"
        hp = dict(OPTIMAL_HPARAMS)
        hp["max_src_len"] = MAX_SRC_LEN_BY_DATASET.get(dataset_name.lower(), hp["max_src_len"])

        dataset_dir = resolve_dataset_dir(data_dir, dataset_name)
        dev_rows = load_json(os.path.join(dataset_dir, "baselines", "_0_fixed_test", "dev.json"))
        test_rows = load_json(os.path.join(dataset_dir, "baselines", "_0_fixed_test", "test.json"))
        clean_src_dir = os.path.join(dataset_dir, "baselines", "_2_Isolate_Data")

        # Phase 1: clean timeline
        run_timeline(
            method=method, src_dir=clean_src_dir,
            out_dir=os.path.join(out_root, dataset_name, "clean", f"seed_{seed}"),
            setting="clean", technique="none", version="clean", n_poisoned=0,
            dataset_name=dataset_name, is_pubmedqa=is_pubmedqa, dev_rows=dev_rows, test_rows=test_rows,
            hp=hp, seed=seed, tokenizer=tokenizer, base_model_name=base_model_name,
            device=device, live_csv_path=live_csv_path,
        )

        # Phase 2: poisoned timelines
        target_versions = pubmedqa_versions if is_pubmedqa else bioasq_versions
        poison_dir = os.path.join(dataset_dir, "PoisonedData")
        if not target_versions:
            target_versions = sorted({int(m.group(1)) for d in os.listdir(poison_dir)
                                      for m in [re.search(r"_v(\d+)$", d)] if m})
        for technique in techniques:
            for k in target_versions:
                v_src_dir = os.path.join(poison_dir, f"_2_Isolate_Data_{technique}_v{k}")
                if not os.path.exists(v_src_dir):
                    continue
                run_timeline(
                    method=method, src_dir=v_src_dir,
                    out_dir=os.path.join(out_root, dataset_name, technique, f"seed_{seed}", f"v{k}"),
                    setting="poisoned", technique=technique, version=f"v{k}", n_poisoned=k,
                    dataset_name=dataset_name, is_pubmedqa=is_pubmedqa, dev_rows=dev_rows, test_rows=test_rows,
                    hp=hp, seed=seed, tokenizer=tokenizer, base_model_name=base_model_name,
                    device=device, live_csv_path=live_csv_path,
                )

        del dev_rows, test_rows
        free_memory()

    print(f"\n[✓ COMPLETE] Benchmark for method '{method}' on seed {seed} finished successfully!\n")


def _count_completed_metrics(output_root: str) -> int:
    n = 0
    if not os.path.exists(output_root):
        return 0
    for _root, _dirs, files in os.walk(output_root):
        if "metrics.json" in files:
            n += 1
    return n


def run_supervisor(output_root: str, max_restarts: int = 50) -> int:
    cmd = [sys.executable, "-u", os.path.abspath(__file__)] + sys.argv[1:] + ["--_worker"]
    os.makedirs(output_root, exist_ok=True)
    log_path = os.path.join(output_root, "supervisor_log.txt")
    restarts, no_progress = 0, 0
    last_progress = _count_completed_metrics(output_root)

    def _log(msg: str):
        line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(f"\n[SUPERVISOR] {msg}", flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    _log(f"Starting worker supervisor (completed splits so far: {last_progress})")
    while True:
        proc = subprocess.Popen(cmd)
        try:
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
            _log("Worker completed all tasks successfully.")
            return 0

        progress = _count_completed_metrics(output_root)
        no_progress = 0 if progress > last_progress else no_progress + 1
        last_progress = progress
        restarts += 1

        if no_progress >= 3 or restarts > max_restarts:
            _log(f"Aborting execution: max restarts reached or no progress made.")
            return rc

        _log(f"Worker fault detected (code {rc}). Resuming from last rolling checkpoint in 20s (attempt {restarts}/{max_restarts})...")
        time.sleep(20)


# CLI & Entry Point
def main():
    parser = argparse.ArgumentParser(description="Continual Learning Comparative Baselines Engine (v2.2)")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the BioASQ2025/ and PubMedQA/ folders")
    parser.add_argument("--techniques", nargs="+", default=TECHNIQUES)
    parser.add_argument("--bioasq_versions", nargs="+", type=int, default=DEFAULT_BIOASQ_VERSIONS)
    parser.add_argument("--pubmedqa_versions", nargs="+", type=int, default=DEFAULT_PUBMEDQA_VERSIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--lr_ft", type=float, default=None, help="Override full-FT learning rate.")
    parser.add_argument("--lr_peft", type=float, default=None, help="Override LoRA learning rate.")
    parser.add_argument("--out_tag", type=str, default=DEFAULT_OUT_TAG,
                        help="Optional suffix for the result folders.")
    parser.add_argument("--no_supervisor", action="store_true", help="Disable self-healing supervisor loop.")
    parser.add_argument("--max_restarts", type=int, default=50)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.lr_ft is not None:
        OPTIMAL_HPARAMS["learning_rate_ft"] = args.lr_ft
    if args.lr_peft is not None:
        OPTIMAL_HPARAMS["learning_rate_peft"] = args.lr_peft

    primary_root = METHOD_OUT_DIRS[args.methods[0]] + args.out_tag

    if not args._worker and not args.no_supervisor and (len(args.methods) > 1 or len(args.seeds) > 1):
        sys.exit(run_supervisor(primary_root, max_restarts=args.max_restarts))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = dict(
        datasets=args.datasets, techniques=args.techniques,
        bioasq_versions=args.bioasq_versions, pubmedqa_versions=args.pubmedqa_versions,
        base_model_name=args.model_name, out_tag=args.out_tag, data_dir=args.data_dir,
    )

    for m in args.methods:
        for s in args.seeds:
            run_method_benchmark(method=m, seed=s, device=device, **common)


if __name__ == "__main__":
    main()