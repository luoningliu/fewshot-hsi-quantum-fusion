#!/usr/bin/env bash
set -euo pipefail

python scripts/hybridsn_spectral_qnn_branch_indian_pines.py \
  --config configs/experiments/hybridsn_spectral_qnn_branch_indian_pines.yaml
