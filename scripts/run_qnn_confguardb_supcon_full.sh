#!/usr/bin/env bash
set -euo pipefail

python scripts/run_supcon_cross_dataset_fewshot.py \
  --datasets indian_pines pavia_university salinas \
  --shots 5 10 \
  --seeds 0 1 2 3 4 \
  --output_dir result/qnn_confguardb_supcon_full_3datasets_5_10shot \
  --gate_context_mode base_confidence_margin \
  --gate_confidence_penalty 0.1 \
  --high_confidence_guard_mode margin_suppression \
  --guard_floor 0.05 \
  --guard_tau 0.35 \
  --guard_temperature 0.08 \
  --monitor macro_f1 \
  --resume
