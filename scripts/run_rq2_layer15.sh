#!/bin/bash
set -e
uv run python3 scripts/rq2_eval_layer.py --layer 15 --max_samples 200 --balanced --output results

set -e
uv run python3 scripts/rq2_eval_layer.py --layer 15 --max_samples 200 --balanced --output results
