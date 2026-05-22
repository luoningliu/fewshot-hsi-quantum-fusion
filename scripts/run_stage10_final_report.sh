#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train --config configs/experiments/stage10_final_report.yaml

