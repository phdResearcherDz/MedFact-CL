#!/usr/bin/env bash
# End-to-end reproduction of MedFact-CL. Run from the repository root.
# Requires: an Ollama server with gemma3:12b, a GPU for the training steps, and the raw
# fact-checking benchmarks for Step 3 (see README.md).
set -euo pipefail

DATA=${DATA:-data}
RAW=${RAW:-raw}
DECISION=${DECISION:-models/decision_model}

# Step 1 - check the released chronological splits
python scripts/prepare_data.py check --data_dir "$DATA"

# Step 2 - poisoned streams (reuses PoisonedData/cache_checkpoint_v2.json when present)
python scripts/poison_data.py --data_dir "$DATA"

# Step 3 - claim-checking cross-encoder
if [ ! -f "$DECISION/config.json" ]; then
  python scripts/train_decision_model.py --scifact "$RAW/scifact" --pubhealth "$RAW/PUBHEALTH" \
      --covidfact "$RAW/RTE-covidfact" --output_dir "$DECISION"
fi

# Step 4 - evidence knowledge graph and FAISS index
python scripts/build_evidence_index.py --data_dir "$DATA"

# Step 5 - trust scores -> data/<dataset>/FactCheckedData/
python scripts/fact_check_streams.py --data_dir "$DATA" --decision_model "$DECISION"

# Step 6 - continual learning
for s in naive ewc replay lora medfact_ewc medfact_replay medfact_lora baselines; do
  python scripts/training/train_${s}.py --data_dir "$DATA"
done
