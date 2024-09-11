
#!/bin/bash

ckpt_path="/pubshare/fwk/ppo_ckpts/2024-08-22-13-25-target3"
root_dir="/home/jovyan/notebook/f-divergence-dpo/outputs/ppo_gt_target3_imdb"
# an array of directory names
# checkpoints=("step-13440" "step-18240" "step-22080" "step-26880" "step-30720" "step-35520" 
# "step-39360" "step-44160" "step-48000" "step-7680" "LATEST" "step-14400" "step-1920" "step-23040" 
# "step-27840" "step-31680" "step-36480" "step-40320" "step-45120" "step-48960" "step-8640" "step-10560" 
# "step-15360" "step-19200" "step-24000" "step-2880" "step-32640" "step-37440" "step-41280" "step-46080" 
# "step-49920" "step-960" "step-11520" "step-16320" "step-20160" "step-24960" "step-28800" "step-33600" 
# "step-3840" "step-42240" "step-47040" "step-5760" "step-9600" "step-12480" "step-17280" "step-21120" 
# "step-25920" "step-29760" "step-34560" "step-38400" "step-43200" "step-4800" "step-6720")
checkpoints=("checkpoint_0000" "best_checkpoint" "checkpoint_0100" "checkpoint_0200" "checkpoint_0300" "checkpoint_0400" "checkpoint_0500" "checkpoint_0600" "checkpoint_0700" "checkpoint_0800" "checkpoint_0900" "checkpoint_1000" "checkpoint_1100" "checkpoint_1200" "checkpoint_1300" "checkpoint_1400")
# iterate through each directory
for (( i=0; i<${#checkpoints[@]}; i+=1 ))
do
  checkpoint=${checkpoints[$i]}
  # perform operations on $checkpoint
  echo "$checkpoint"
  # for example, if you want to list the contents of each directory
  # uncomment the following line
  # ls "$checkpoint"
# done | xargs -I {} -P 5 srun --gres=gpu:1 -c 6 --mem 60G -p general --exclude=g002,g005,g006,g007,g008,g009  python metrics/imdb/imdb_eval_metrics.py --checkpoint $ckpt_path/{}/policy.pt --divergence reverse_kl
# done | xargs -I {} -P 1 srun --gres=gpu:1 -c 6 --mem 20G -p general python metrics/imdb/imdb_eval_metrics.py --sft true --checkpoint $ckpt_path/{}/policy.pt --divergence reverse_kl
done | CUDA_VISIBLE_DEVICES=3 ROOT_DIR=$root_dir xargs -I {} -P 1 python metrics/imdb/imdb_eval_metrics.py --sft true --checkpoint $ckpt_path/{}/model.safetensors --divergence reverse_kl

# done | xargs -I {} -P 1 srun --gres=gpu:1 -c 6 --mem 20G -p general bash -c 'CUDA_VISIBLE_DEVICES=5,6 python metrics/imdb/imdb_eval_metrics.py --sft true --checkpoint /pubshare/fwk/dpo_cache/jovyan/imdb_dpo_gpt2_large_2024-08-21_02-16-51_047361/{}/policy.pt --divergence reverse_kl'
# srun --gres=gpu:1 -c 6 --mem 20G -p general python metrics/imdb/imdb_eval_metrics.py --checkpoint $ckpt_path/step-960/policy.pt --divergence reverse_kl

# export ckpt_path="/pubshare/fwk/dpo_cache/jovyan/imdb_dpo_gpt2_large_2024-08-21_02-16-51_047361"
# CUDA_VISIBLE_DEVICES=3 ROOT_DIR="/home/jovyan/notebook/f-divergence-dpo/outputs/ppo_gt_target3_imdb" python metrics/imdb/imdb_eval_metrics.py --sft true --checkpoint /pubshare/fwk/ppo_ckpts/2024-08-22-13-25-target3/checkpoint_0100/model.safetensors --divergence reverse_kl