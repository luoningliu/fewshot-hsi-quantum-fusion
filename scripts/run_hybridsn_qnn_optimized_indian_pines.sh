#!/usr/bin/env bash
set -euo pipefail

python scripts/optimize_hybridsn_qnn_indian_pines.py \
  --config configs/experiments/hybridsn_qnn_optimized_indian_pines.yaml
