import os
os.environ['PYTHONUNBUFFERED'] = 'True'

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import glob
import os
import numpy as np
from scipy.special import softmax

def load_model(model_path, weights_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict['state'])
    model = model.to(device)
    model.eval()
    return model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_path = "EleutherAI/pythia-2.8b" #input("Enter the path to your base model: ")
    model_path = "gpt2-large"
    # weights_path = "/path/to/dpo_forward_kl_pythia28_beta0.1/LATEST/policy.pt" #input("Enter the path to your trained model's weights: ")
    # weights_paths = glob.glob("/path/to/HH/pythia28/*/LATEST/policy.pt")
    weights_paths = glob.glob('/pubshare/fwk/dpo_cache/jovyan/anthropic_dpo_reverse_kl_pythia28_hh0.1_2024-08-20_15-18-13_795249/LATEST/policy.pt')
    weights_paths = glob.glob('/pubshare/fwk/dpo_cache/jovyan/imdb_dpo_alpha_divergence_0.3_gpt2_large_hh0.01_10_epochs_2024-09-17_14-34-51_014994/LATEST/policy.pt')
    for weights_path in weights_paths:
        # weights_path =  #input("Enter the path to your trained model's weights: ")
        saved_path = os.path.join(os.path.dirname(weights_path), "hf_model")
        ckpt_path = os.path.join(saved_path, "pytorch_model*.bin")
        if len(glob.glob(ckpt_path)) > 0:
            continue
        tokenizer_name = model_path #input("Enter the tokenizer name: ")
        model = load_model(model_path, weights_path, device)
        print('Model loaded.')
        model.save_pretrained(saved_path)
        print("Converted HF Model Saved to: {}".format(saved_path))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(saved_path)
        print("Converted HF Model Tokenizer Saved to: {}".format(saved_path))

