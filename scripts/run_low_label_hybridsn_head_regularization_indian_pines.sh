#!/usr/bin/env bash
set -euo pipefail

python scripts/low_label_hybridsn_head_regularization_indian_pines.py \
  --config configs/experiments/low_label_hybridsn_head_regularization_indian_pines.yaml
