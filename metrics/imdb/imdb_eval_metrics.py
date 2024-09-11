import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
from safetensors.torch import load_file
from tqdm import tqdm
import os
import argparse
import tree
from pathlib import Path
def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]



# create the top-level parser
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint')
parser.add_argument('--divergence', type=str, help='what kind of divergence')
parser.add_argument('--alpha', type=float, default=0.5, help='what kind of divergence')
parser.add_argument('--sft', type=bool, default=False, help='is sft')

args = parser.parse_args()
parent_dir = os.path.dirname(args.checkpoint)
# root_dit = '/home/jovyan/notebook/f-divergence-dpo/outputs/sft_imdb'
root_dit = os.environ['ROOT_DIR']
if not os.path.exists(root_dit):
    os.makedirs(root_dit)
step_idxs = args.checkpoint.split('/')[-2]
print(f'Checkpoint: {args.checkpoint}')
f_divergence = args.divergence

print('Saving to:')
print(os.path.join(root_dit, f'{f_divergence}_{args.alpha}_{step_idxs}.txt'))
print('*' * 80  + '\n')
path = os.path.join(root_dit, f'{f_divergence}_{args.alpha}_{step_idxs}.txt')
if not os.path.exists(path):
    print(f"The file does not exist. Continue running your process.")
    # Insert the code to run your process here
else:
    print(f"The file {path} exists. Exit the process.")
    exit() # Use sys.exit() if this doesn't work


assert f_divergence in args.checkpoint or args.sft, 'f_divergence must be in the checkpoint path'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

def get_f_divergence(f_divergence, alpha=0.5):
    if f_divergence == 'reverse_kl':
        f = lambda x: -torch.log(x)
        f_prime_one = -1.
    elif f_divergence == 'forward_kl':
        f = lambda x: x * torch.log(x)
        f_prime_one = 1.
    elif f_divergence == 'jsd':
        f = lambda x: x * torch.log(x) - (x+1) * torch.log((x+1)/2)
        f_prime_one = 0. 
    elif f_divergence == 'alpha_divergence':
        assert alpha > 0, 'alpha must be positive'
        assert alpha < 1, 'alpha must be less than 1'
        f = lambda x: (1.0 / (alpha * (alpha - 1))) * (torch.pow(x, alpha) - alpha * x + alpha - 1)
        f_prime_one = 0.
    else:
        raise NotImplementedError
    return f, f_prime_one
torch.cuda.empty_cache()
# 检查是否有可用的 GPU
print('CUDA_VISBLE_DEVICES:', os.getenv('CUDA_VISIBLE_DEVICES', 'not set'))
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # 转换为 GB
        gpu_free_memory = torch.cuda.memory_reserved(i) / (1024 ** 3)  # 转换为 GB
        gpu_allocated_memory = torch.cuda.memory_allocated(i) / (1024 ** 3)  # 转换为 GB
        gpu_available_memory = gpu_memory - gpu_allocated_memory

        print(f"GPU {i}: {gpu_name}")
        print(f"  Total memory: {gpu_memory:.2f} GB")
        print(f"  reserved memory: {gpu_free_memory:.2f} GB") 
        print(f"  Free memory: {gpu_available_memory:.2f} GB")
else:
    print("No GPUs available.")

set_seed(42)

f, f_prime_one = get_f_divergence(f_divergence, 1-args.alpha)

# Load your trained model
state_dict_path = args.checkpoint #.cache/chaoqi/imdb_dpo_reverse_kl_gpt2_large_hh0.1_2023-07-10_20-46-25_643455/LATEST/policy.pt'  # insert your model path here


tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side="left"
if tokenizer.pad_token is None:
    tokenizer.pad_token=tokenizer.eos_token

state_dict_path = Path(state_dict_path).resolve()
model = AutoModelForCausalLM.from_pretrained('gpt2-large')
print(state_dict_path)
# breakpoint()
if state_dict_path.exists():
    # model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu'))['state'])
    if str(state_dict_path).endswith('.safetensors'):
        state_dict = load_file(state_dict_path, device='cpu')
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu'))['state'], strict=False)
elif (state_dict_path.parent / 'hf_model').exists():
    model = AutoModelForCausalLM.from_pretrained(str(state_dict_path.parent / 'hf_model'))
else:
    raise FileNotFoundError(f"Model not loaded correctly")
# import ipdb; ipdb.set_trace()
model.to('cuda')

# Load reference model
# ref_model_name = '.cache/chaoqi/imdb_dpo_gpt2_large_2023-07-10_16-45-15_446529/LATEST/policy.pt'  # this can be changed to another model if needed
ref_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
# ref_tokenizer.truncation_side = "right"
ref_tokenizer.padding_side="left"
if ref_tokenizer.pad_token is None:
    ref_tokenizer.pad_token=tokenizer.eos_token
ref_model = AutoModelForCausalLM.from_pretrained("gpt2-large")
# ref_model.load_state_dict(torch.load(ref_model_name, map_location=torch.device('cpu'))['state'])
ref_model.to('cuda')
# import ipdb; ipdb.set_trace()

sentiment_fn = pipeline(
    "sentiment-analysis",
    "siebert/sentiment-roberta-large-english",
    top_k=2,
    truncation=True,
    batch_size=64,
    device=model.device  # specify the device id here
)
# Load the imdb dataset
imdb_test = load_dataset("imdb", split="test")

# Preprocess the dataset
eval_prompts = [" ".join(review.split()[:4]) for review in imdb_test["text"]]
inputs = tokenizer(eval_prompts, return_tensors='pt', truncation=True, padding=True)

# Prepare for batching
dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], )
print(len(dataset))
data_loader = DataLoader(dataset, batch_size=64)

# Prepare for entropy and divergence calculations
total_entropy = 0
total_f_divergence = 0
total_num_items = 0
total_reward = 0

DEBUG_0 = False
DEBUG_1 = False
DEBUG_2 = False
with torch.no_grad():
    for batch_input_ids, batch_attention_mask in tqdm(data_loader):
        # Generate samples from the pretrained model
        # import ipdb; ipdb.set_trace()
        batch_input_ids = batch_input_ids.cuda()
        batch_attention_mask = batch_attention_mask.cuda()
        # with torch.no_grad():
        generated_ids = model.generate(batch_input_ids, attention_mask=batch_attention_mask, do_sample=True, max_new_tokens=60, pad_token_id=tokenizer.pad_token_id)
        
        # Get log probabilities for the generated samples
        
        # with torch.no_grad():
        if True:
            if DEBUG_0:
                import ipdb; ipdb.set_trace()
            model_inputs = tokenizer(tokenizer.batch_decode(generated_ids), return_tensors='pt', padding=True)
            model_inputs = tree.map_structure(lambda x: x.to(model.device), model_inputs)
            # 传入 labels，自动计算loss
            model_outputs = model(**model_inputs, labels=model_inputs['input_ids'])
            model_log_probs = model_outputs.logits.softmax(dim=-1).log()

            ref_inputs = ref_tokenizer(tokenizer.batch_decode(generated_ids), return_tensors='pt', padding=True)
            ref_inputs = tree.map_structure(lambda x: x.to(ref_model.device), ref_inputs)
            ref_outputs = ref_model(**ref_inputs, labels=ref_inputs['input_ids'])
            # (batch_size, seq_len, vocab_size)
            ref_log_probs = ref_outputs.logits.softmax(dim=-1).log()
        
        generated_ids = model_inputs['input_ids']
        #(batch_size, seq_len)
        attention_mask = (generated_ids != tokenizer.eos_token_id).float()
        if DEBUG_0:
            import ipdb; ipdb.set_trace()
        # Calculate entropy
        try:
            batch_entropy = (-((model_log_probs.exp() * model_log_probs).sum(-1)) * attention_mask).sum() / attention_mask.sum()
            total_entropy += batch_entropy.cpu().item() * len(batch_input_ids)
        except:
            import ipdb; ipdb.set_trace()

        # Calculate f-divergence
        if DEBUG_1:
            import ipdb; ipdb.set_trace()
        # attention_mask: (batch_size, seq_len, 1)
        #log(p/q)
        log_ratio = (model_log_probs - ref_log_probs) * attention_mask.unsqueeze(-1)
        ratio = torch.exp(-log_ratio)  # q/p
        f_ratio = f(ratio) # f(q/p) * p
        # f_divergence = f_ratio - f_prime_one * (ratio - 1)
        # f_divergence = f_divergence.sum(1).mean()  # sum over tokens, mean over batch
        f_divergence = f_ratio * torch.exp(model_log_probs) * attention_mask.unsqueeze(-1)
        f_divergence = f_divergence.sum()  # sum over tokens
        total_f_divergence += f_divergence.item()
        # .mean().item() * len(batch_input_ids)
        
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # if DEBUG_2:
        # import ipdb; ipdb.set_trace()
        sentiments = sentiment_fn(generated_texts)
        sentiment_scores = [get_positive_score(sentiment) for sentiment in sentiments]
        # sentiment_scores = [sentiment_fn(text)[0][0]['score'] for text in generated_texts]
        total_reward += sum(sentiment_scores)
        

        total_num_items += len(batch_input_ids)

# Compute averages
average_entropy = total_entropy / total_num_items
average_f_divergence = total_f_divergence / total_num_items
average_reward = total_reward / total_num_items
print(f'Averaged entropy: {average_entropy}')
print(f'Averaged f-divergence: {average_f_divergence}')
print(f'Averaged reward: {average_reward}')

with open(os.path.join(root_dit, f'{args.divergence}_{args.alpha}_{step_idxs}.txt'), 'w') as f:
    f.write(f'Averaged entropy: {average_entropy}\n')
    f.write(f'Averaged f-divergence: {  average_f_divergence}\n')   
    f.write(f'Averaged reward: {average_reward}\n')


