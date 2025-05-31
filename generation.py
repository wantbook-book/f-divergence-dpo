import os
os.environ['PYTHONUNBUFFERED'] = 'True'

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from scipy.special import softmax
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)

from preference_datasets import get_dataset


def load_model(model_path, weights_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict['state'])
    model = model.to(device)
    model.eval()
    return model

def calculate_entropy(logits):
    # Compute the probabilities from the logits
    probabilities = softmax(logits, axis=-1)

    # Compute the entropy from the probabilities
    entropy = -np.sum(probabilities * np.log2(probabilities), axis=-1)

    return entropy.mean()

def calculate_self_bleu(text):
    words = text.split()
    reference = words[:-1]
    candidate = words[1:]
    score = sentence_bleu([reference], candidate)
    return score

def interact_model(model, tokenizer_name, device, temperature=1, top_p=1):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("You can start the conversation now:")
    while True:
        input_text = input("User: ")
        encoded_input = tokenizer.encode_plus(input_text, return_tensors='pt')
        encoded_input = encoded_input.to(device)

        responses = []
        for _ in range(10):
            if temperature != 1:
                outputs = model.generate(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'], max_length=512, temperature=temperature, do_sample=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
            else:
                # outputs = model.generate(encoded_input, max_length=200, top_p=top_p, do_sample=True, return_dict_in_generate=True)
                outputs = model.generate(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'], max_length=512, top_p=top_p, do_sample=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)

            response = tokenizer.decode(outputs.sequences[:, encoded_input['input_ids'].shape[-1]:][0], skip_special_tokens=True)
            entropy = 0  # calculate_entropy(outputs.scores.cpu().detach().numpy())
            self_bleu = 0  # calculate_self_bleu(response)

            responses.append((response, entropy, self_bleu))

        for i, (response, entropy, self_bleu) in enumerate(responses):
            print(f"AI {i+1}: ", response)
            print(f"Entropy: {entropy}, Self-BLEU: {self_bleu}")
        import ipdb; ipdb.set_trace()

def generate_from_data_prompt(model, tokenizer_name, device, temperature=1, top_p=1):
    silent = False
    split = 'test'
    name = 'imdb'
    # 字典，key是prompt，value是chosen啥的response等
    dataset = get_dataset(name, split, silent=silent, cache_dir=get_local_dir(['.cache']))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    f = open('responses.txt', 'w')
    for i, prompt in enumerate(dataset.keys()):
        if i > 10:
            break
        # input_text = input("User: ")
        input_text = prompt
        encoded_input = tokenizer.encode_plus(input_text, return_tensors='pt')
        encoded_input = encoded_input.to(device)

        if temperature != 1:
            outputs = model.generate(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'], max_length=512, temperature=temperature, do_sample=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
        else:
            # outputs = model.generate(encoded_input, max_length=200, top_p=top_p, do_sample=True, return_dict_in_generate=True)
            outputs = model.generate(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'], max_length=512, top_p=top_p, do_sample=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs.sequences[:, encoded_input['input_ids'].shape[-1]:][0], skip_special_tokens=True)
        entropy = 0  # calculate_entropy(outputs.scores.cpu().detach().numpy())
        self_bleu = 0  # calculate_self_bleu(response)

        f.write(f"Prompt: {prompt}\n")
        f.write(f"Response: {prompt+response}\n")
        # f.write(f"Entropy: {entropy}, Self-BLEU: {self_bleu}\n")
        f.write("="*80 + "\n")
        # import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #input("Enter the path to your base model: ")
    model_path = "gpt2-large" 
    #input("Enter the path to your trained model's weights: ")
    weights_path = "/home/jovyan/notebook/f-divergence-dpo/trainning_results/imdb_sft_gpt2_large_2024-08-09_02-51-15_299597/LATEST/policy.pt" 
    # weights_path = ".cache/chaoqi/anthropic_dpo_reverse_kl_pythia28_hh_2023-06-26_17-06-05_245154/LATEST/policy.pt"   # reverse KL
    #input("Enter the tokenizer name: ")
    tokenizer_name = "gpt2-large" 
    model = load_model(model_path, weights_path, device)
    print('Model loaded.')
    method = "top-p" #input("Please select a sampling method - (T)emperature or (Top-p): ")
    if method.lower() == 't':
        temperature = float(input("Enter the temperature value (default=1): "))
        generate_from_data_prompt(model, tokenizer_name, device, temperature=temperature)
        # interact_model(model, tokenizer_name, device, temperature=temperature)
    elif method.lower() == 'top-p':
        top_p = 0.95 #float(input("Enter the top-p value (default=1): "))
        generate_from_data_prompt(model, tokenizer_name, device, top_p=top_p)
        # interact_model(model, tokenizer_name, device, top_p=top_p)
    else:
        print("Invalid selection.")

