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
with open('data/activationPatching_2hop_false.json', 'r') as f:
    alternate_COTData = json.load(f)
# %%
import json
with open('data/activationPatching_2hop.json', 'r') as f:
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
MODEL_PATH = '/home/models/vicuna-7b'
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
save_dir = f"results/activationPatching/{noise_index}/accuracy"
os.makedirs(save_dir, exist_ok=True)
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


# %%
print("RUNNING ORIGINAL LOGITS")
original_logits, cache = runCacheActivationPatching(model, input_ids)

# %%
print("RUNNING NOISE LOGITS")
alternate_logits, alternate_cache = runCacheActivationPatching(model, alternate_input_ids)

# %%
import pickle
with open(f'results/activationPatching/{noise_index}/kl_div_COT.pickle', 'rb') as f:
    patched_head = pickle.load(f)
patched_head = patched_head.detach().cpu()
# %%
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
standardized_head = standard_scaler.fit_transform(patched_head.cpu())

# %%
data = []
for layer in range(len(standardized_head)):
    for head in range(len(standardized_head[layer])):
        data.append({"layer" : layer, "head" : head, "value" : standardized_head[layer][head]})
# %%
sorted_data = sorted(data, key=lambda x: x['value'], reverse=True)
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

data = patched_head.detach().cpu().numpy()
data_flat = data.reshape(-1)
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
count_acc_range = []
from tqdm import tqdm
for i in tqdm(range(1023, -1, -10)):
    heads_to_remove =sorted_data[i + 1:]
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
# %%
# from tqdm import tqdm
# count_acc_range = []
# for threshold_range in tqdm(threshold_ranges):

#     print(threshold_range)
#     count = 0
#     list_fwd_hooks = []
#     for layer in range(len(patched_head)):
#         for head in range(len(patched_head[layer])):
#             if(patched_head[layer][head] > threshold_range[1] or patched_head[layer][head] < threshold_range[0]):
#                 continue
#             else:
#                 list_fwd_hooks.append((utils.get_act_name("z", layer, "attn"), partial(patch_head_vector_avg, head_index=head, alternate_cache=alternate_cache)))
#                 # print(layer,head)
#                 # print(patched_head[layer][head])
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
#     if(args.combined):
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
