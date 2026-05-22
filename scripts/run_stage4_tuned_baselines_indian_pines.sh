#!/usr/bin/env bash
set -euo pipefail

python -m src.training.tuned_baselines --config configs/experiments/stage4_tuned_baselines_indian_pines.yaml

