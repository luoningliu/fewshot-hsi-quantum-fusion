#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train --config configs/experiments/stage6_quantum_sweep.yaml

