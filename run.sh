#!/usr/bin/env bash
set -euo pipefail

METHOD="$1"; shift
DATASETS=("moe_scaling_law" "parallel_scaling_law" "sae_scaling_law" "sft_scaling_law" "sparsity_scaling_law" "vocab_scaling_law")
EXTRA_ARGS=("$@")

# Query scaling law IDs from the dataset's LAW_REGISTRY

echo "Method:  $METHOD"
echo "Extra:   ${EXTRA_ARGS[*]:-}"
echo "---"

for dataset in ${DATASETS[@]}; do
    echo "Launching $dataset ..."
    python -m benchmark.main \
        --dataset "$dataset" \
        --method "$METHOD" \
        --fitter lbfgsb \
        --repeat 30 \
        --workers 4 \
        --n-restarts 32 \
        --no-plot \
        "${EXTRA_ARGS[@]}" 
done
