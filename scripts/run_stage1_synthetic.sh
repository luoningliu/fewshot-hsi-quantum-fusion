#!/usr/bin/env bash
set -euo pipefail

python scripts/make_synthetic_lcz42.py
python -m src.datasets.preprocessing --config configs/experiments/stage1_data_protocol_synthetic.yaml

