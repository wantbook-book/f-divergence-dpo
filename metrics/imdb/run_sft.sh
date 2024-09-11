
#!/bin/bash

ckpt_path="/pubshare/fwk/dpo_cache/jovyan/imdb_dpo_gpt2_large_2024-08-21_02-16-51_047361"
root_dir="/home/jovyan/notebook/f-divergence-dpo/outputs/sft_imdb"
# an array of directory names
# checkpoints=("step-13440" "step-18240" "step-22080" "step-26880" "step-30720" "step-35520" 
# "step-39360" "step-44160" "step-48000" "step-7680" "LATEST" "step-14400" "step-1920" "step-23040" 
# "step-27840" "step-31680" "step-36480" "step-40320" "step-45120" "step-48960" "step-8640" "step-10560" 
# "step-15360" "step-19200" "step-24000" "step-2880" "step-32640" "step-37440" "step-41280" "step-46080" 
# "step-49920" "step-960" "step-11520" "step-16320" "step-20160" "step-24960" "step-28800" "step-33600" 
# "step-3840" "step-42240" "step-47040" "step-5760" "step-9600" "step-12480" "step-17280" "step-21120" 
# "step-25920" "step-29760" "step-34560" "step-38400" "step-43200" "step-4800" "step-6720")
checkpoints=("step-128" "step-256" "step-384" "step-512" "step-640" "step-768" "step-896" "step-1024" "step-1152" "step-1280" "step-1408" "step-1536" "step-1664" "step-1792" "step-1920" "step-2048" "step-2176" "step-2304" "step-2432" "step-2560" "step-2688" "step-2816" "step-2944" "step-3072" "step-3200" "step-3328" "step-3456" "step-3584" "step-3712" "step-3840" "step-3968" "step-4096" "step-4224" "step-4352" "step-4480" "step-4608" "step-4736" "step-4864" "step-4992" "step-5120" "step-5248" "step-5376" "step-5504" "step-5632" "step-5760" "step-5888" "step-6016" "step-6144" "step-6272" "step-6400" "step-6528" "step-6656" "step-6784" "step-6912" "step-7040" "step-7168" "step-7296" "step-7424" "step-7552" "step-7680" "step-7808" "step-7936" "step-8064" "step-8192" "step-8320" "step-8448" "step-8576" "step-8704" "step-8832" "step-8960" "step-9088" "step-9216" "step-9344" "step-9472" "step-9600" "step-9728" "step-9856" "step-9984" "step-10112" "step-10240" "step-10368" "step-10496" "step-10624" "step-10752" "step-10880" "step-11008" "step-11136" "step-11264" "step-11392" "step-11520" "step-11648" "step-11776" "step-11904" "step-12032" "step-12160" "step-12288" "step-12416" "step-12544" "step-12672" "step-12800" "step-12928" "step-13056" "step-13184" "step-13312" "step-13440" "step-13568" "step-13696" "step-13824" "step-13952" "step-14080" "step-14208" "step-14336" "step-14464" "step-14592" "step-14720" "step-14848" "step-14976" "step-15104" "step-15232" "step-15360" "step-15488" "step-15616" "step-15744" "step-15872" "step-16000" "step-16128" "step-16256" "step-16384" "step-16512" "step-16640" "step-16768" "step-16896" "step-17024" "step-17152" "step-17280" "step-17408" "step-17536" "step-17664" "step-17792" "step-17920" "step-18048" "step-18176" "step-18304" "step-18432" "step-18560" "step-18688" "step-18816" "step-18944" "step-19072" "step-19200" "step-19328" "step-19456" "step-19584" "step-19712" "step-19840" "step-19968" "step-20096" "step-20224" "step-20352" "step-20480" "step-20608" "step-20736" "step-20864" "step-20992" "step-21120" "step-21248" "step-21376" "step-21504" "step-21632" "step-21760" "step-21888" "step-22016" "step-22144" "step-22272" "step-22400" "step-22528" "step-22656" "step-22784" "step-22912" "step-23040" "step-23168" "step-23296" "step-23424" "step-23552" "step-23680" "step-23808" "step-23936" "step-24064" "step-24192" "step-24320" "step-24448" "step-24576" "step-24704" "step-24832" "step-24960" "step-25088" "step-25216" "step-25344" "step-25472" "step-25600" "step-25728" "step-25856" "step-25984" "step-26112" "step-26240" "step-26368" "step-26496" "step-26624" "step-26752" "step-26880" "step-27008" "step-27136" "step-27264" "step-27392" "step-27520" "step-27648" "step-27776" "step-27904" "step-28032" "step-28160" "step-28288" "step-28416" "step-28544" "step-28672" "step-28800" "step-28928" "step-29056" "step-29184" "step-29312" "step-29440" "step-29568" "step-29696" "step-29824" "step-29952" "step-30080" "step-30208" "step-30336" "step-30464" "step-30592" "step-30720" "step-30848" "step-30976" "step-31104" "step-31232" "step-31360" "step-31488" "step-31616" "step-31744" "step-31872" "step-32000" "step-32128" "step-32256" "step-32384" "step-32512" "step-32640" "step-32768" "step-32896" "step-33024" "step-33152" "step-33280" "step-33408" "step-33536" "step-33664" "step-33792" "step-33920" "step-34048" "step-34176" "step-34304" "step-34432" "step-34560" "step-34688" "step-34816" "step-34944" "step-35072" "step-35200" "step-35328" "step-35456" "step-35584" "step-35712" "step-35840" "step-35968" "step-36096" "step-36224" "step-36352" "step-36480" "step-36608" "step-36736" "step-36864" "step-36992" "step-37120" "step-37248" "step-37376" "step-37504" "step-37632" "step-37760" "step-37888" "step-38016" "step-38144" "step-38272" "step-38400" "step-38528" "step-38656" "step-38784" "step-38912" "step-39040" "step-39168" "step-39296" "step-39424" "step-39552" "LATEST")
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
done | CUDA_VISIBLE_DEVICES=1 ROOT_DIR=$root_dir xargs -I {} -P 1 python metrics/imdb/imdb_eval_metrics.py --sft true --checkpoint $ckpt_path/{}/policy.pt --divergence reverse_kl

# done | xargs -I {} -P 1 srun --gres=gpu:1 -c 6 --mem 20G -p general bash -c 'CUDA_VISIBLE_DEVICES=5,6 python metrics/imdb/imdb_eval_metrics.py --sft true --checkpoint /pubshare/fwk/dpo_cache/jovyan/imdb_dpo_gpt2_large_2024-08-21_02-16-51_047361/{}/policy.pt --divergence reverse_kl'
# srun --gres=gpu:1 -c 6 --mem 20G -p general python metrics/imdb/imdb_eval_metrics.py --checkpoint $ckpt_path/step-960/policy.pt --divergence reverse_kl

# export ckpt_path="/pubshare/fwk/dpo_cache/jovyan/imdb_dpo_gpt2_large_2024-08-21_02-16-51_047361"
# CUDA_VISIBLE_DEVICES=1 python metrics/imdb/imdb_eval_metrics.py --sft true --checkpoint $ckpt_path/step-128/policy.pt --divergence reverse_kl