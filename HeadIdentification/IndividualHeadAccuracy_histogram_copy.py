# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from jaxtyping import Float, Int
from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

# %%
import circuitsvis as cv
# Testing that the library works
# cv.examples.hello("Neel")

# %%
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# %%
torch.set_grad_enabled(False)
# %%
import json
with open('data/activationPatching_llama2_false.json', 'r') as f:
    alternate_COTData = json.load(f)
# %%
import json
with open('data/activationPatching_llama2.json', 'r') as f:
    COTData = json.load(f)
# %%
import argparse
parser = argparse.ArgumentParser()
# python3 IndividualHeadAccuracy.py --noiseIndex 0 --device "cuda:1" --numExamples 10 --alteranteExamples 10
parser.add_argument("-noiseIndex", "--noiseIndex", help = "noiseIndex")
parser.add_argument("-device", "--device", help = "device")
parser.add_argument("-numExamples", "--numExamples", help = "numExamples")
parser.add_argument("-alternateExamples", "--alternateExamples", help = "alternateExamples")

# alternateExamples
# parser.add_argument("--generate", action="store_true", help = "generate")
# parser.add_argument("-modelPath", "--modelPath", help = "modelPath")
# parser.add_argument("--combined", action="store_true", help = "combined")
# parser.add_argument("-combine_type", "--combine_type", help = "combine_type")

args = parser.parse_args()
# %%
from utilsFile.fewShotConstants import TemplateCOT_fictional, TemplateCOT_false

# %%
# device = "cuda:2"
device = args.device if torch.cuda.is_available() else "cpu"

# %%

from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from utilsFile.loadModel import loadTransformerLensModel, runCacheActivationPatching
MODEL_PATH = '/home/models/Llama-2-7b-hf'
# modelPath='/home/models/vicuna-7b'
model, tokenizer = loadTransformerLensModel(MODEL_PATH)
model = model.to(device)
# %%
noise_index = int(args.noiseIndex)
number_of_examples = int(args.numExamples)
alternate_examples = int(args.alternateExamples)
# noise_index = 0
# number_of_examples = 5
# alternate_examples = 5
import os
save_dir = f"results/copy/combined/{noise_index}/accuracy_prob"
os.makedirs(save_dir, exist_ok=True)
print(save_dir)
# %%

print(f"Noise index: {noise_index}")
print(f"Number of examples: {number_of_examples}")
prompts = [TemplateCOT_fictional.format(data['prompt']) + data[f'response_{noise_index}'] for data in COTData][:number_of_examples]
labels = [(data[f'response_{noise_index + 1}'].replace(data[f'response_{noise_index}'], "").strip()) for data in COTData][:number_of_examples]
alternate_prompts = [TemplateCOT_false.format(data['prompt']) + data[f'response_{noise_index}'] for data in alternate_COTData][:alternate_examples]
alternate_labels = [data['label'] for data in alternate_COTData][:alternate_examples]

# %%
input_ids = []
max_length = 0
for prompt in prompts:
    finalPrompt = prompt
    encoded_prompt = tokenizer.encode(finalPrompt, add_special_tokens=False, return_tensors="pt")
    max_length = max(max_length, encoded_prompt.size()[1])
    input_ids.append(encoded_prompt[0])


# %%
alternate_max_length = 0
alternate_input_ids = []
for prompt in alternate_prompts:
    finalPrompt = prompt
    encoded_prompt = tokenizer.encode(finalPrompt, add_special_tokens=False, return_tensors="pt")
    alternate_max_length = max(alternate_max_length, encoded_prompt.size()[1])
    alternate_input_ids.append(encoded_prompt[0])

# %%
max_seq_len = max(max_length, alternate_max_length)

# %%
pad_id = tokenizer.bos_token_id

# %%
# padd all input_ids to max length from left
for i in range(len(input_ids)):
    input_ids[i] = [pad_id] * (max_seq_len - len(input_ids[i])) + input_ids[i].detach().numpy().tolist()
input_ids = torch.tensor(input_ids)

# %%
for i in range(len(alternate_input_ids)):
    alternate_input_ids[i] = [pad_id] * (max_seq_len - len(alternate_input_ids[i])) + alternate_input_ids[i].detach().numpy().tolist()
alternate_input_ids = torch.tensor(alternate_input_ids)
# %%

    
# %%
def logit_accuracy(logits, patched_logits):
    next_token_logits = logits[:, -1, :]
    next_token_ids = torch.argmax(next_token_logits, dim=-1)
    next_token_ids = [[tokens.item()] for tokens in next_token_ids]
    accuracy = 0
    next_tokens = model.tokenizer.batch_decode(next_token_ids)

    next_token_patched_logits = patched_logits[:, -1, :]
    next_token_patched_ids = torch.argmax(next_token_patched_logits, dim=-1)
    next_token_patched_ids = [[tokens.item()] for tokens in next_token_patched_ids]
    next_patched_tokens = model.tokenizer.batch_decode(next_token_patched_ids)
    # print(next_tokens)
    # print(next_tokens)
    # print(next_patched_tokens)
    for i in range(len(next_tokens)):
        next_token = next_tokens[i].strip()
        next_patched_token = next_patched_tokens[i].strip()
        
        if(next_token == next_patched_token):
            accuracy += 1
    return accuracy / len(next_tokens)

import pickle
with open(f'results/copy/combined/combined_matrix_attn_prob.pkl', 'rb') as f:
    patched_head = pickle.load(f)
# patched_head = patched_head.cpu().numpy()
# %%
print("RUNNING ORIGINAL LOGITS")
original_logits, cache = runCacheActivationPatching(model, input_ids)

# %%
print("RUNNING NOISE LOGITS")
alternate_logits, alternate_cache = runCacheActivationPatching(model, alternate_input_ids)

# %%

# patched_head = patched_head.detach().cpu().numpy()
# %%
# from sklearn.preprocessing import StandardScaler
# 
# standard_scaler = StandardScaler()
# standardized_head = standard_scaler.fit_transform(patched_head.cpu())
data_flat = patched_head.reshape(-1)
import numpy as np
import matplotlib.pyplot as plt

data_range = np.max(data_flat) - np.min(data_flat)
bin_width = 2 * (np.percentile(data_flat, 75) - np.percentile(data_flat, 25)) / (len(data_flat) ** (1/3))
num_bins = int(data_range / bin_width)
# %%
hist, bin_edges = np.histogram(data_flat, bins=num_bins)

# %%
max_range_idx = np.argmax(hist)
max_range = (bin_edges[max_range_idx], bin_edges[max_range_idx + 1])
# %%
threshold_ranges = []

# threshold_ranges.append([bin_edges[max_range_idx], bin_edges[max_range_idx]])
for i in range(0, 10):
    left_idx = max_range_idx - i
    right_idx = max_range_idx + i
    if(left_idx < 0):
        left_idx = 0
    if(right_idx >= len(hist)):
        right_idx = len(hist) - 1
    # threshold_ranges.append([bin_edges[left_idx +  1], bin_edges[right_idx]])
    threshold_ranges.append([bin_edges[left_idx], bin_edges[right_idx]])

    # left_idx = max_range_idx - i +  1
    # right_idx = max_range_idx + i
    # if(left_idx < 0):
    #     left_idx = 0
    # if(right_idx >= len(hist)):
    #     right_idx = len(hist) - 1
    # # threshold_ranges.append([bin_edges[left_idx +  1], bin_edges[right_idx]])
    # threshold_ranges.append([bin_edges[left_idx], bin_edges[right_idx]])
    # if 0 <= left_idx < len(hist) and 0 <= right_idx < len(hist):
# for i in range(20):
    # threshold_ranges.append([bin_edges[max_range_idx-i], bin_edges[max_range_idx + i]])
# %%

# %%
def patch_head_vector(
    corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
    hook, 
    head_index, 
    clean_cache):
    # print(hook.name, head_index, corrupted_head_vector.shape)
    corrupted_head_vector[:, :, head_index, :] = clean_cache[hook.name][:, :, head_index, :]
    return corrupted_head_vector
# %%
def patch_head_vector_avg(
        clean_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
        hook,
        head_index,
        alternate_cache):
    mean_alternate_cache = torch.mean(alternate_cache[hook.name][:, :, head_index, :], dim=0)
    zero_tensor_cache = torch.zeros_like(mean_alternate_cache)
    clean_head_vector[:, :, head_index, :] = mean_alternate_cache
    return clean_head_vector

import numpy as np
import matplotlib.pyplot as plt

def generate_text(input_ids, max_new_tokens):
        original_input_ids = input_ids.clone()
        while(max_new_tokens > 0):
            patched_logits = model.run_with_hooks(
                    input_ids,
                    fwd_hooks = list_fwd_hooks,
                    return_type="logits"
                )
            next_token_id = torch.argmax(patched_logits[:, -1, :], dim=-1)
            next_token = model.tokenizer.convert_ids_to_tokens(next_token_id)[0]

            if(next_token == '<0x0A>'):
                break
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(1)], dim=-1)
            max_new_tokens -= 1
        generated_text = model.tokenizer.decode(input_ids[0][original_input_ids.size(1):], skip_special_tokens=True)
        return generated_text

# %%
threshold_range = [-100, 0]
count = 0
list_fwd_hooks = []
for layer in range(len(patched_head)):
    for head in range(len(patched_head[layer])):
        if(patched_head[layer][head] > threshold_range[1] or patched_head[layer][head] < threshold_range[0]):
            continue
        else:
            list_fwd_hooks.append((utils.get_act_name("z", layer, "attn"), partial(patch_head_vector_avg, head_index=head, alternate_cache=alternate_cache)))
#                 # print(layer,head)
#                 # print(patched_head[layer][head])
            count += 1
print(f'Heads with activation outside threshold range, {threshold_range}: {count}')
print("Number of heads removed or patched: ", count)

patched_logits = model.run_with_hooks(
            input_ids, 
            fwd_hooks = list_fwd_hooks, 
            return_type="logits"
        )

patched_acc = logit_accuracy(original_logits, patched_logits)

print(f"Accuracy of next token after patching {count} heads: {patched_acc}")
# %%
sorted_data = []
for layer in range(len(patched_head)):
    for head in range(len(patched_head[layer])):
        sorted_data.append({'layer' : layer, 'head' : head, 'value' : patched_head[layer][head]})

sorted_data = sorted(sorted_data, key=lambda x: x['value'])
# %%
count_acc_range = []
from tqdm import tqdm
for i in tqdm(range(11, 600, 25)):
    heads_to_remove =sorted_data[:i]
    print(f"Value range: {heads_to_remove[0]['value']} - {heads_to_remove[-1]['value']}")
    print("Number of heads in circuit :", 1024 - len(heads_to_remove))
    list_fwd_hooks = []
    for layer in range(32):
        for head in range(32):
            for data in heads_to_remove:
                if(data['layer'] == layer and data['head'] == head):
                    # list_fwd_hooks.append([0])
                    list_fwd_hooks.append((utils.get_act_name("z", layer, "attn"), partial(patch_head_vector_avg, head_index=head, alternate_cache=alternate_cache)))
    print("Number of heads patched :", len(list_fwd_hooks))
    patched_logits = model.run_with_hooks(
                input_ids, 
                fwd_hooks = list_fwd_hooks, 
                return_type="logits"
            )
    patched_acc = logit_accuracy(original_logits, patched_logits)
    print(f"Accuracy of next token after patching {len(heads_to_remove)} heads: {patched_acc}")
    count_acc_range.append({'circuit' : 1024 - len(heads_to_remove), 'accuracy' : patched_acc})
    
    with open(f'{save_dir}/circuit_results.pickle', 'wb') as f:
        pickle.dump(count_acc_range,f)  
    if(patched_acc <= 0):
        break 
# %%
# from tqdm import tqdm
# count_acc_range = []
# for threshold_range in tqdm(threshold_ranges):

#     # print(threshold_range)
#     count = 0
#     list_fwd_hooks = []
#     for layer in range(len(patched_head)):
#         for head in range(len(patched_head[layer])):
#             if(patched_head[layer][head] > threshold_range[1] or patched_head[layer][head] < threshold_range[0]):
#                 continue
#             else:
#                 list_fwd_hooks.append((utils.get_act_name("z", layer, "attn"), partial(patch_head_vector_avg, head_index=head, alternate_cache=alternate_cache)))
# #                 # print(layer,head)
# #                 # print(patched_head[layer][head])
#                 count += 1
#     print(f'Heads with activation outside threshold range, {threshold_range}: {count}')
#     print("Number of heads removed or patched: ", count)

#     patched_logits = model.run_with_hooks(
#                 input_ids, 
#                 fwd_hooks = list_fwd_hooks, 
#                 return_type="logits"
#             )

#     patched_acc = logit_accuracy(original_logits, patched_logits)
    
#     print(f"Accuracy of next token after patching {count} heads: {patched_acc}")
#     count_acc_range.append({'count' : count, 'accuracy' : patched_acc, 'range' : threshold_range})
#     import pickle
#     with open(f'{save_dir}/circuit_results_histogram.pickle', 'wb') as f:
#         pickle.dump(count_acc_range,f) 
#     if(patched_acc <= 0):
#         break
#         with open(f'iteration_3_2hop_base/head_COT_{noise_index}_combined_{args.combine_type}_circuit_results.pickle', 'wb') as f:
#             pickle.dump(count_acc_range,f)
#     else:
#         with open(f'iteration_3_2hop_base/head_COT_{noise_index}_circuit_results.pickle', 'wb') as f:
#             pickle.dump(count_acc_range,f)     

    # if(True):
        # generated_texts = []
        # for input_id in input_ids:
            # generated_texts.append(generate_text(input_id.unsqueeze(0).to(device), 100))

        # accuracy = 0
        # for i in range(len(generated_texts)):
            # if(labels[i] in generated_texts[i]):
                # accuracy += 1
        # print(f"Accuracy of generated text after patching {count} heads: {accuracy / len(generated_texts)}")

# %%
