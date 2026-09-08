#!/bin/bash
# 轻量入口：调用 Python 生产者-消费者 cached dataset 构建器。
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN=${PYTHON_BIN:-python3}
CONFIG=${CONFIG:-./cached_dataset_config.example.toml}
exec "$PYTHON_BIN" ./distributed_cached_dataset.py run --config "$CONFIG" "$@"
