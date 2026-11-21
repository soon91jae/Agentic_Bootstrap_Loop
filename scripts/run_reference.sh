#!/usr/bin/env bash
set -euo pipefail
aboot_data_dir="${1:-data/odd}"
aboot_run_dir="${2:-runs}"
for aboot_mode in sft grpo soft hybrid; do
  python -m aboot.cli run --data "$aboot_data_dir" \
    --config "configs/${aboot_mode}.json" --output "${aboot_run_dir}/${aboot_mode}_oracle" \
    --oracle-simulation
done
