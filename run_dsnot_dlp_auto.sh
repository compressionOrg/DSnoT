#!/usr/bin/env bash
set -euo pipefail
set -x
# 测试DSnoT + DLP 70%稀疏率下的性能
# Usage:
#   bash run_dsnot_dlp_auto.sh [model_name_or_path]
# Example:
#   bash run_dsnot_dlp_auto.sh meta-llama/Llama-2-7b-hf

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

MODEL=${1:-meta-llama/Llama-2-7b-hf}
SPARSITY_RATIO=${SPARSITY_RATIO:-0.7}
NSAMPLES=${NSAMPLES:-128}
SEED=${SEED:-0}

python main.py \
  --model "${MODEL}" \
  --prune_method DSnoT_dlp_auto \
  --initial_method wanda \
  --sparsity_ratio "${SPARSITY_RATIO}" \
  --sparsity_type unstructured \
  --nsamples "${NSAMPLES}" \
  --seed "${SEED}" \
  --max_cycle_time 50 \
  --update_threshold 0.1 \
  --pow_of_var_regrowing 1 \
  --alpha 0.15 \
  --auto_alpha \
  --alpha_min 0.0 \
  --alpha_max 0.3 \
  --alpha_tolerance 0.025 \
  --alpha_max_iter 10 \
  --save_model "llama2_7b_sparsity_${SPARSITY_RATIO}_DSnoT_dlp_auto" \
  --eval_zero_shot \
  2>&1 | tee "dsnot_dlp_auto_${SPARSITY_RATIO}.log"

set +x
