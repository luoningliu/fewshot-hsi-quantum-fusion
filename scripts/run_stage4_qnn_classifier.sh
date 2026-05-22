#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train --config configs/experiments/stage4_qnn_classifier.yaml

