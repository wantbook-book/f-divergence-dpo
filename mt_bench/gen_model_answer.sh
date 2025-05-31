# model_id 用来标识结果的
CUDA_VISIBLE_DEVICES="5" python3 mt_bench/gen_model_answer.py \
    --model-path EleutherAI/pythia-2.8b \
    --state-dict-path /pubshare/fwk/dpo_cache/jovyan/anthropic_dpo_reverse_kl_pythia28_hh0.1_2024-08-20_15-18-13_795249/LATEST/policy.pt \
    --model-id pythia28 \
    --bench-name mt_bench \
    --answer-file output_answers.jsonl \
    --max-new-token 1024 \
    --num-choices 1 \
    --num-gpus-per-model 1 \
    --num-gpus-total 1 \
    --max-gpu-memory "64GB"
