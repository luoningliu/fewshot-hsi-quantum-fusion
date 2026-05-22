#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train --config configs/experiments/stage8_feature_analysis.yaml

