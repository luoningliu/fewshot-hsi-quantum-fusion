#!/usr/bin/env bash
set -euo pipefail

python scripts/make_synthetic_hsi.py
python -m src.datasets.hsi_preprocessing --config configs/experiments/stage1_data_protocol_synthetic_hsi.yaml

