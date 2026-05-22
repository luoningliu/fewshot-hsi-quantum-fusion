#!/usr/bin/env bash
set -euo pipefail

python -m src.training.residual_qnn_full --config configs/experiments/stage4_residual_qnn_full_indian_pines.yaml

