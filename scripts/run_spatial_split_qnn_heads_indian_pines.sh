#!/usr/bin/env bash
set -euo pipefail

python scripts/spatial_split_qnn_heads_indian_pines.py \
  --config configs/experiments/spatial_split_qnn_heads_indian_pines.yaml
