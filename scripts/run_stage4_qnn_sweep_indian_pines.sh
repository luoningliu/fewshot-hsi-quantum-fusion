#!/usr/bin/env bash
set -euo pipefail

python -m src.training.qnn_sweep --config configs/experiments/stage4_qnn_sweep_indian_pines.yaml

