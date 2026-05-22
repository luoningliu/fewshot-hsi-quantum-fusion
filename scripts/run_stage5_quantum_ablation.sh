#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train --config configs/experiments/stage5_quantum_ablation.yaml

