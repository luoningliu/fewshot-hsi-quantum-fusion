#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train --config configs/experiments/stage9_complexity.yaml

