set -x 

export CUDA_VISIBLE_DEVICES=3

python main.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --prune_method DSnoT \
    --initial_method wanda \
    --sparsity_ratio 0.7 \
    --sparsity_type unstructured \
    --max_cycle_time 50 \
    --update_threshold 0.1 \
    --pow_of_var_regrowing 1 \
    --save_model "llama2_7b_sparsity_0.7_DSnoT" \
    --eval_zero_shot \
    2>&1 | tee llama2_7b_eval.log

set +x 