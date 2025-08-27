#!/bin/bash
set -e

uv run python3 scripts/rq2_eval_layer.py --layer full --max_samples 200 --balanced --model_dir "$MODEL_DIR_OVERRIDE" --output results

set -e
uv run python3 scripts/rq2_eval_layer.py --layer 28 --max_samples 200 --balanced --output results
