# 12:57pm, June 26, 2023
######## training for dpo using forward_kl but sft is done on hh and shp. ########
# model.archive=.cache/chaoqi/imdb_dpo_gpt2_large_2023-07-10_16-45-15_446529/LATEST/policy.pt 
BETA=0.01
EPOCHS=10

values=(0.1 0.3 0.5 0.7 0.9)
alpha=0.1

CUDA_VISIBLE_DEVICES=2 python -u train.py \
     n_epochs=$EPOCHS \
     lr=1.5e-6 \
     model=gpt2_large \
     datasets=[imdb] \
     loss=dpo \
     fsdp_port=49155 \
     loss.divergence=alpha_divergence \
     loss.alpha=$alpha \
     loss.beta=$BETA \
     exp_name=imdb_dpo_alpha_divergence_${alpha}_gpt2_large_hh${BETA}_${EPOCHS}_epochs \
     gradient_accumulation_steps=2 \
     batch_size=64 \
     eval_batch_size=32 \
     trainer=BasicTrainer \
     sample_during_eval=false \
     model.fsdp_policy_mp=bfloat16 \
     eval_every=128