#!/usr/bin/env bash
set -euo pipefail

python -m src.datasets.hsi_preprocessing --config configs/experiments/stage1_data_protocol.yaml

