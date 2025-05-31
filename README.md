# $\gamma$-DPO

##  Requirements
To get started, please install the required libraries first:
```
ipykernel==6.23.1
numpy==1.24.3
tokenizers==0.13.3
torch==2.0.1
tqdm==4.65.0
transformers==4.29.2
datasets==2.12.0
beautifulsoup4==4.12.2
wandb==0.15.3
hydra-core==1.3.2
tensor-parallel==1.2.4
```



## How to Run?

1. $\gamma$-DPO

```bash
bash scripts/imdb/run_gamma_dpo.sh
```

2. N-DPO

```bash
bash scripts/imdb/run_n_dpo.sh
```

3. DPO

```bash
bash scripts/imdb/run_reverse_kl.sh
```
