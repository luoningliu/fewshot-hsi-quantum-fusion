#!/usr/bin/env bash
set -euo pipefail

python -m src.training.reupload_multiobs_qnn_full --config configs/experiments/stage4_reupload_multiobs_qnn_full_indian_pines.yaml

