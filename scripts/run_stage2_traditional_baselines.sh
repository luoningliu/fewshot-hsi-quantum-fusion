#!/usr/bin/env bash
set -euo pipefail

python -m src.training.traditional_baselines --config configs/experiments/stage2_traditional_baselines.yaml

