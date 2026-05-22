#!/usr/bin/env bash
set -euo pipefail

python scripts/tune_hybridsn_indian_pines.py \
  --config configs/experiments/hybridsn_tuning_indian_pines.yaml
