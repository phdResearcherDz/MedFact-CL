#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Naive sequential fine-tuning baseline.

Usage:
    python scripts/train_naive.py --data_dir data --output_root results/Naive
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
import random
import argparse
import datetime
import traceback
import subprocess
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
    
    # Calculate token budget dynamically
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


# Live Console Telemetry Box
def print_metric_box(header_text: str, metrics: Dict[str, Any], duration_sec: float = 0.0):
    acc = metrics.get("accuracy", 0.0)
    f1 = metrics.get("f1_macro", 0.0)
    prec = metrics.get("precision_macro", 0.0)
    rec = metrics.get("recall_macro", 0.0)
    auroc = metrics.get("auroc")
    brier = metrics.get("brier_score")

    auroc_str = f"{auroc:.4f}" if isinstance(auroc, (int, float)) else "N/A"
    brier_str = f"{brier:.4f}" if isinstance(brier, (int, float)) else "N/A"

    line_w = 78
    print("\n" + "┌" + "─" * (line_w - 2) + "┐")
    print(f"│  {header_text:<{line_w - 5}} │")
    print("├" + "─" * (line_w - 2) + "┤")
    print(f"│  Accuracy: {acc:.4f} ({acc*100:5.2f}%)   |  Macro F1: {f1:.4f} ({f1*100:5.2f}%)                │")
    print(f"│  Precision: {prec:.4f}          |  Recall:   {rec:.4f}                          │")
    print(f"│  AUROC:    {auroc_str:<14}  |  Brier:    {brier_str:<14}  [Time: {duration_sec:.1f}s] │")
    print("└" + "─" * (line_w - 2) + "┘\n", flush=True)

LIVE_CSV_HEADER = [
    "timestamp", "seed", "dataset", "setting", "technique", "version",
    "split", "total_splits", "accuracy", "f1_macro", "precision_macro",
    "recall_macro", "auroc", "brier_score", "duration_sec"
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
    os.replace(tmp_path, csv_path)


# Crash-Safe Rolling Checkpoints
def save_rolling_checkpoint(model, tokenizer, ckpt_dir: str, completed_split: int):
    os.makedirs(ckpt_dir, exist_ok=True)
    marker_file = os.path.join(ckpt_dir, "progress.json")
    if os.path.exists(marker_file):
        try:
            os.remove(marker_file)
        except Exception:
            pass

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    save_json({
        "completed_split": int(completed_split),
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

    # AUROC & Brier Score Calculation
    try:
        if len(set(gold_ids)) >= 2:
            if not is_pubmedqa:
                clean_yes = np.clip(np.nan_to_num(np.array(yes_probs, dtype=float), nan=0.5), 1e-6, 1.0 - 1e-6)
                auroc = float(roc_auc_score(gold_ids, clean_yes))
                brier = float(brier_score_loss(gold_ids, clean_yes))
            else:
                clean_all = np.clip(np.nan_to_num(np.array(all_probs_list, dtype=float), nan=1.0/len(cand_names)), 1e-6, 1.0 - 1e-6)
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


# Single Split Fine-Tuning Step with Semantic Margin Loss
def _tokenize_rows(
    tokenizer: AutoTokenizer, 
    rows: List[Dict[str, Any]], 
    is_pubmedqa: bool, 
    max_src_len: int, 
    max_tgt_len: int
) -> List[Dict[str, List[int]]]:
    prompts = [format_prompt(r.get("question"), r.get("context"), is_pubmedqa=is_pubmedqa,
                             tokenizer=tokenizer, max_src_len=max_src_len) for r in rows]
    targets = [normalize_label(r.get("answer"), is_pubmedqa=is_pubmedqa) for r in rows]
    enc = tokenizer(prompts, truncation=True, max_length=max_src_len)
    lab = tokenizer(text_target=targets, truncation=True, max_length=max_tgt_len)
    return [
        {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": lab["input_ids"][i]}
        for i in range(len(rows))
    ]

@torch.no_grad()
def compute_dev_loss(model, dev_loader, device) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in dev_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        loss = model(**batch).loss
        n_tok = int((batch["labels"] != -100).sum().item())
        if torch.isfinite(loss):
            total_loss += float(loss.item()) * n_tok
            total_tokens += n_tok
    return total_loss / max(1, total_tokens) if total_tokens else float("inf")

def compute_step_loss(
    model: AutoModelForSeq2SeqLM,
    batch: Dict[str, torch.Tensor],
    cand_t: torch.Tensor,
    margin: float = 0.20,
    margin_lambda: float = 0.35
) -> torch.Tensor:
    """Combines Seq2Seq Cross-Entropy with first-step logit candidate margin separation."""
    outputs = model(**batch)
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

def train_single_split(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    train_rows: List[Dict[str, Any]],
    dev_rows: List[Dict[str, Any]],
    is_pubmedqa: bool,
    out_split_dir: str,
    hparams: Dict[str, Any],
    seed: int = 42,
    micro_batch_size: int = 4,
    eval_batch_size: int = 16,
    max_tgt_len: int = 8,
    max_grad_norm: float = 1.0
) -> AutoModelForSeq2SeqLM:
    hp = {**OPTIMAL_HPARAMS, **hparams}
    device = next(model.parameters()).device
    seed_all(seed)
    os.makedirs(out_split_dir, exist_ok=True)
    shutil.rmtree(os.path.join(out_split_dir, "_temp_run"), ignore_errors=True)

    max_src_len = int(hp["max_src_len"])
    train_feats = _tokenize_rows(tokenizer, train_rows, is_pubmedqa, max_src_len, max_tgt_len)
    dev_feats = _tokenize_rows(tokenizer, dev_rows, is_pubmedqa, max_src_len, max_tgt_len)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest", pad_to_multiple_of=8, label_pad_token_id=-100)
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

    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    skipped_steps = 0
    patience = int(hp["patience"])

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        t_ep = time.time()
        run_loss, run_n = 0.0, 0
        group_nonfinite = False

        pbar = tqdm(train_loader, total=n_micro, leave=False, dynamic_ncols=True, desc=f"  epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            group_idx = step // grad_accum
            group_len = min(grad_accum, n_micro - group_idx * grad_accum)

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = compute_step_loss(model, batch, cand_t, margin=float(hp["semantic_margin"]), margin_lambda=float(hp["margin_lambda"]))

            if not torch.isfinite(loss):
                group_nonfinite = True
            else:
                (loss / group_len).backward()
                run_loss += float(loss.item())
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
                pbar.set_postfix(loss=f"{run_loss / max(1, run_n):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        pbar.close()

        dev_loss = compute_dev_loss(model, dev_loader, device)
        train_loss = run_loss / max(1, run_n)
        improved = dev_loss < best_loss
        if improved:
            best_loss, best_epoch, bad_epochs = dev_loss, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 5), "dev_loss": round(dev_loss, 5),
            "improved": improved, "peak_vram_gb": round(peak_gb, 2), "seconds": round(time.time() - t_ep, 1)
        })
        print(f"  epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | dev_loss {dev_loss:.4f}"
              f"{'  *best*' if improved else ''} | peak VRAM {peak_gb:.2f} GB | {time.time() - t_ep:.1f}s", flush=True)

        if bad_epochs >= patience:
            print(f"  [early stop] no dev improvement for {patience} epochs -> restoring epoch {best_epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_json({
        "best_epoch": best_epoch, "best_dev_loss": round(best_loss, 5),
        "skipped_nonfinite_steps": skipped_steps, "epochs": history
    }, os.path.join(out_split_dir, "training_log.json"))

    del optimizer, scheduler, best_state, train_loader, dev_loader, collator, train_feats, dev_feats
    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    return model


# Continual Learning Runners
def run_clean_baseline_seed(
    dataset_name: str,
    seed: int,
    base_dir: str,
    out_dir_seed: str,
    live_csv: str,
    model_name: str,
    device: torch.device,
    hparams: Dict[str, Any]
) -> Dict[str, Any]:
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
    if ckpt_path and 1 <= ckpt_split <= last_completed_idx:
        for s_i in range(ckpt_split + 1, last_completed_idx + 1):
            completed_splits.pop(f"s{s_i}", None)
        print(f"\n[⚡ RESUMING] Clean {dataset_name} (Seed {seed}): Resuming at split s{ckpt_split + 1}...")
        current_model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).to(device)
        start_split_idx = ckpt_split + 1
    else:
        current_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        start_split_idx = 1

    all_metrics = copy.deepcopy(completed_splits)

    for s_idx in range(start_split_idx, total_splits + 1):
        s_name = f"s{s_idx}"
        split_out = os.path.join(out_dir_seed, s_name)
        print(f"\n[Seed {seed} | Clean {dataset_name}] Training Split {s_idx}/{total_splits} ({s_name}.json)...")
        train_rows = load_json(os.path.join(split_dir, f"{s_name}.json"))

        t0 = time.time()
        current_model = train_single_split(
            model=current_model, tokenizer=tokenizer, train_rows=train_rows, dev_rows=dev_rows,
            is_pubmedqa=is_pubmedqa, out_split_dir=split_out, hparams=hp, seed=seed
        )
        eval_metrics, predictions_log = evaluate_fixed_test(
            model=current_model, tokenizer=tokenizer, test_rows=test_rows, is_pubmedqa=is_pubmedqa,
            device=device, max_src_len=int(hp["max_src_len"])
        )
        duration = time.time() - t0

        all_metrics[s_name] = eval_metrics
        save_json(eval_metrics, os.path.join(split_out, "metrics.json"))
        save_json(predictions_log, os.path.join(split_out, "predictions.json"))
        save_rolling_checkpoint(current_model, tokenizer, rolling_ckpt_dir, s_idx)

        print_metric_box(f"Split {s_name} Clean Test Results (Seed {seed} | {dataset_name})", eval_metrics, duration)
        append_live_split_csv(live_csv, {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": seed, "dataset": dataset_name, "setting": "clean", "technique": "none",
            "version": "v0", "split": s_name, "total_splits": total_splits,
            "accuracy": eval_metrics["accuracy"], "f1_macro": eval_metrics["f1_macro"],
            "precision_macro": eval_metrics["precision_macro"], "recall_macro": eval_metrics["recall_macro"],
            "auroc": eval_metrics.get("auroc", ""), "brier_score": eval_metrics.get("brier_score", ""),
            "duration_sec": round(duration, 1)
        })
        save_json(all_metrics, summary_file)

    del current_model
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

    poison_dir = os.path.join(base_dir, "PoisonedData")
    dev_rows = load_json(os.path.join(base_dir, "baselines", "_0_fixed_test", "dev.json"))
    test_rows = load_json(os.path.join(base_dir, "baselines", "_0_fixed_test", "test.json"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    snapshots_dir = os.path.join(out_dir_tech_seed, "_poisoned_snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    v_dirs = [d for d in os.listdir(poison_dir) if d.startswith(f"_2_Isolate_Data_{technique}_v")]
    v_nums = sorted([int(re.search(r'v(\d+)', d).group(1)) for d in v_dirs])
    target_versions = [v for v in v_nums if v in versions] if versions else v_nums

    master_summary: Dict[str, Dict[str, Any]] = {}

    for k in target_versions:
        v_src_dir = os.path.join(poison_dir, f"_2_Isolate_Data_{technique}_v{k}")
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
                start_split_idx = ckpt_split + 1
            elif last_completed_idx >= k and snap_k_path:
                for s_i in range(k + 1, last_completed_idx + 1):
                    completed_splits.pop(f"s{s_i}", None)
                print(f"\n[⚡ RESUMING] Version v{k} (Seed {seed}): Boundary split s{k} complete. Resuming from snapshot...")
                current_model = AutoModelForSeq2SeqLM.from_pretrained(snap_k_path).to(device)
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
                    start_split_idx = k
                else:
                    current_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
                    start_split_idx = 1
            else:
                current_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
                start_split_idx = 1

        v_metrics = copy.deepcopy(completed_splits)

        for s_idx in range(start_split_idx, total_splits + 1):
            s_name = f"s{s_idx}"
            split_out = os.path.join(v_out_dir, s_name)
            split_type_str = "POISONED" if s_idx <= k else "CLEAN"
            print(f"\n>>> [Seed {seed} | {dataset_name} | {technique} | v{k}] Split {s_idx}/{total_splits} ({split_type_str}) <<<")

            train_rows = load_json(os.path.join(v_src_dir, f"{s_name}.json"))
            t0 = time.time()
            current_model = train_single_split(
                model=current_model, tokenizer=tokenizer, train_rows=train_rows, dev_rows=dev_rows,
                is_pubmedqa=is_pubmedqa, out_split_dir=split_out, hparams=hp, seed=seed
            )
            eval_metrics, predictions_log = evaluate_fixed_test(
                model=current_model, tokenizer=tokenizer, test_rows=test_rows, is_pubmedqa=is_pubmedqa,
                device=device, max_src_len=int(hp["max_src_len"])
            )
            duration = time.time() - t0

            if s_idx == k:
                print(f"  [💾] Saving continuation snapshot for split s{k} -> {snap_k_dir}")
                save_rolling_checkpoint(current_model, tokenizer, snap_k_dir, k)

            v_metrics[s_name] = eval_metrics
            save_json(eval_metrics, os.path.join(split_out, "metrics.json"))
            save_json(predictions_log, os.path.join(split_out, "predictions.json"))
            save_rolling_checkpoint(current_model, tokenizer, rolling_v_ckpt_dir, s_idx)

            print_metric_box(f"Split {s_name} [{split_type_str}] (Seed {seed} | {dataset_name} | {technique} | v{k})", eval_metrics, duration)
            append_live_split_csv(live_csv, {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seed": seed, "dataset": dataset_name, "setting": "poisoned", "technique": technique,
                "version": f"v{k}", "split": s_name, "total_splits": total_splits,
                "accuracy": eval_metrics["accuracy"], "f1_macro": eval_metrics["f1_macro"],
                "precision_macro": eval_metrics["precision_macro"], "recall_macro": eval_metrics["recall_macro"],
                "auroc": eval_metrics.get("auroc", ""), "brier_score": eval_metrics.get("brier_score", ""),
                "duration_sec": round(duration, 1)
            })
            save_json(v_metrics, summary_file)

        shutil.rmtree(rolling_v_ckpt_dir, ignore_errors=True)
        master_summary[f"v{k}"] = v_metrics
        del current_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return master_summary


# Statistical Aggregations (Mean +/- Std Across Seeds)
def compute_multiseed_aggregations(output_root: str, dataset_name: str, seeds: List[int]):
    print(f"\n{'='*75}\n COMPUTING MULTI-SEED STATISTICAL AGGREGATIONS: {dataset_name}\n{'='*75}")
    ds_root = os.path.join(output_root, dataset_name)
    num_splits = 12 if dataset_name.lower() == "bioasq2025" else 6

    clean_runs: Dict[int, Dict[str, Any]] = {}
    poison_runs: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {t: {} for t in TECHNIQUES}

    for s in seeds:
        cf = os.path.join(ds_root, "clean", f"seed_{s}", "all_splits_metrics.json")
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
            split_agg_rows.append(aggregate_split("Clean", "None", "v0", f"s{i}", clean_runs))
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
        r = aggregate_overall("Clean", "None", "v0", clean_runs)
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
    cmd = [sys.executable, "-u", os.path.abspath(__file__)] + sys.argv[1:] + ["--_worker"]
    os.makedirs(args.output_root, exist_ok=True)
    log_path = os.path.join(args.output_root, "supervisor_log.txt")
    restarts, no_progress = 0, 0
    last_progress = _count_completed_splits(args.output_root)

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
            _log("Worker completed all splits successfully.")
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
    parser = argparse.ArgumentParser(description="Multi-Seed Semantic Continual Learning Pipeline")
    parser.add_argument("--datasets", nargs="+", default=["PubMedQA", "BioASQ2025"], choices=["BioASQ2025", "PubMedQA"])
    parser.add_argument("--techniques", nargs="+", default=["all"])
    parser.add_argument("--mode", default="all", choices=["clean", "poisoned", "all"])
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the BioASQ2025/ and PubMedQA/ folders")
    
    # Version filtering: universal or per-dataset
    parser.add_argument("--versions", nargs="+", type=int, default=DEFAULT_VERSIONS, help="Universal version filter applied across datasets")
    parser.add_argument("--pubmedqa_versions", nargs="+", type=int, default=DEFAULT_PUBMEDQA_VERSIONS, help="Specific version numbers for PubMedQA (e.g. 1 2 3)")
    parser.add_argument("--bioasq_versions", nargs="+", type=int, default=DEFAULT_BIOASQ_VERSIONS, help="Specific version numbers for BioASQ2025 (e.g. 1 2 3)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_root", type=str, default="results/Naive")
    parser.add_argument("--aggregate_only", action="store_true")

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
    parser.add_argument("--hparams_file", type=str, default=None)

    # Supervisor / Worker
    parser.add_argument("--no_supervisor", action="store_true")
    parser.add_argument("--max_restarts", type=int, default=50)
    parser.add_argument("--keep_snapshots", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if not args._worker and not args.no_supervisor and not args.aggregate_only:
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
        "margin_lambda": args.margin_lambda
    }

    if args.hparams_file and os.path.exists(args.hparams_file):
        active_hparams.update(load_json(args.hparams_file))
        print(f"[+] Loaded custom hyperparameters from {args.hparams_file}")

    print("\n" + "="*80)
    print(" MULTI-SEED SEMANTIC CONTINUAL LEARNING ENGINE")
    print(f" Model        : {args.model_name}")
    print(f" Device       : {device}")
    print(f" Datasets     : {args.datasets}")
    print(f" Seeds        : {args.seeds}")
    print(f" Output Root  : {args.output_root}")
    print(f" Margin / Lmb : {active_hparams['semantic_margin']} / {active_hparams['margin_lambda']}")
    print("="*80 + "\n")

    if args.aggregate_only:
        for ds in args.datasets:
            compute_multiseed_aggregations(args.output_root, ds, args.seeds)
        return

    chosen_techs = TECHNIQUES if "all" in args.techniques else args.techniques

    for ds in args.datasets:
        ds_base = resolve_dataset_dir(args.data_dir, ds)
        ds_out = os.path.join(args.output_root, ds)
        ds_versions = get_target_versions_for_dataset(args, ds)

        print(f"[*] Dataset: {ds} | Resolved Directory: {ds_base}")
        if ds_versions is not None:
            print(f"[*] Target Versions for {ds}: {ds_versions}")

        for s_idx, current_seed in enumerate(args.seeds, start=1):
            seed_all(current_seed)
            print(f"\n{'#'*80}\n [Seed {current_seed} ({s_idx}/{len(args.seeds)})] Executing Conditions for {ds}\n{'#'*80}")

            if args.mode in ["clean", "all"]:
                run_clean_baseline_seed(
                    dataset_name=ds, seed=current_seed, base_dir=ds_base,
                    out_dir_seed=os.path.join(ds_out, "clean", f"seed_{current_seed}"),
                    live_csv=live_csv, model_name=args.model_name, device=device, hparams=active_hparams
                )

            if args.mode in ["poisoned", "all"]:
                for tech in chosen_techs:
                    run_poisoned_versions_seed(
                        dataset_name=ds, technique=tech, seed=current_seed, base_dir=ds_base,
                        out_dir_tech_seed=os.path.join(ds_out, tech, f"seed_{current_seed}"),
                        live_csv=live_csv, model_name=args.model_name, device=device, hparams=active_hparams,
                        versions=ds_versions, keep_snapshots=args.keep_snapshots
                    )

        compute_multiseed_aggregations(args.output_root, ds, args.seeds)

    print("\n" + "="*80)
    print(" ALL CONTINUAL LEARNING EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()