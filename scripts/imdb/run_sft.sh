# 12:57pm, June 26, 2023
######## repeat training using sft on hh only
# ulimit -n 64000; CUDA_VISIBLE_DEVICES="0,1,2" python -u train.py model=gpt2_large datasets=[imdb] loss=sft exp_name=imdb_dpo_gpt2_large gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false
ulimit -n 64000; CUDA_VISIBLE_DEVICES="5,6" python -u train.py model=gpt2_large datasets=[imdb] loss=sft exp_name=imdb_dpo_gpt2_large lr=1e-6 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 eval_every=128 trainer=BasicTrainer sample_during_eval=false