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
import pickle

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

# %%
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from torch import Tensor
import io

# %%
torch.set_grad_enabled(False)

# %%
device = 'cuda:0'

# %%
def loadTransformerLensModel(modelPath):
    tokenizer = AutoTokenizer.from_pretrained(modelPath)
    hf_model = AutoModelForCausalLM.from_pretrained(modelPath, low_cpu_mem_usage=True)
    model = HookedTransformer.from_pretrained("meta-llama/Llama-2-7b-hf", hf_model=hf_model, device='cpu', fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer)

    return model, tokenizer

# %%
MODEL_PATH = "meta-llama/Llama-2-7b-hf"
model, tokenizer = loadTransformerLensModel(MODEL_PATH)
model = model.to(device)


import argparse
parser = argparse.ArgumentParser()
# python3 IndividualHeadAccuracy.py --noiseIndex 0 --device "cuda:1" --numExamples 10 --alteranteExamples 10
parser.add_argument("-noiseIndex", "--noiseIndex", help = "noiseIndex")
# parser.add_argument("-start", "--start", help = "start")
# parser.add_argument("-end", "--end", help = "end")
# parser.add_argument("-device", "--device", help = "device")
parser.add_argument("-scaleFactor", "--scaleFactor", help = "scaleFactor")
# parser.add_argument("-type", "--type", help = "type")

args = parser.parse_args()

noise_index = int(args.noiseIndex)
scaleFactor = float(args.scaleFactor)
# type = args.type
# %%
noise_index = 0
number_of_examples = 100
# %%
# import pickle
# with open(f'writing_heads_{noise_index}.pkl', 'rb') as f:
#     writing_heads = pickle.load(f)

import pickle
with open(f"/home/t-josingh/How-To-Think-Step-by-Step/InformationFlow/result/5/commonHeads_ABC.pickle", "rb") as f:
    common_heads = pickle.load(f)

# %%
# writing_heads[0][:10]

# %%
# top_k_heads = []
# bottom_k_heads = []
# k = 5
# filtered_heads = []
# for heads in common_heads:
#     filtered_heads = []
#     for head in heads:
#         # if(head['layer'] > 16):
#         filtered_heads.append(head)
#     for head in filtered_heads[:k]:
#         top_k_heads.append((head['layer'], head['head']))
    
#     for head in filtered_heads[-k:]:
#         bottom_k_heads.append((head['layer'], head['head']))

# %%
# top_k_heads = set(top_k_heads)
# bottom_k_heads = set(bottom_k_heads)
common_heads = set(common_heads)

# %%
# len(top_k_heads), len(bottom_k_heads)

# %%
import json
with open('/home/t-josingh/How-To-Think-Step-by-Step/data/activationPatching_llama2_clean.json', 'r') as f:
    COTData = json.load(f)




# %%
def getInputId(prompt):
    encoded_prompt =tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    return encoded_prompt

# %%
def getCacheLogits(input_id):
    patched_cache = {}
    list_fwd_hooks = []
    def storeHookCache(value, hook):
        patched_cache[hook.name] = torch.from_numpy(value.detach().cpu().numpy())
    for layer in range(32):

        list_fwd_hooks.append((utils.get_act_name("z", layer, "attn"), storeHookCache))
            
    patched_logits = model.run_with_hooks(
            input_id, 
            fwd_hooks = list_fwd_hooks, 
            return_type="logits"
        )
    return patched_logits, patched_cache

# %%
import sys
sys.path.append('../')
from utilsFile.fewShotConstants import TemplateCOT_fictional, TemplateCOT_false

# %%
print(f"Noise index: {noise_index}")
print(f"Number of examples: {number_of_examples}")
print(f"Scaled Factor: {scaleFactor}")
prompts = [TemplateCOT_fictional.format(data['prompt']) + data[f'response_{noise_index}'] for data in COTData][:number_of_examples]


# %%
input_id = getInputId(prompts[0])
logit, cache = getCacheLogits(input_id)

# %%
token_X_id = torch.argmax(logit[:, -1, :], dim=1)[0]
# initial_layer_head = stage_I(-1, token_X_id, 32)

# %%
probability = torch.nn.functional.softmax(logit[:, -1, :], dim=1)[0, token_X_id].item()

# %%
# scaleFactor = 0

# %%
def scale_attention_heads(
        clean_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
        hook,
        head_index,
        scale_number):
    clean_head_vector[:, :, head_index, :] = torch.mul(clean_head_vector[:, :, head_index, :], scale_number)
    return clean_head_vector

# %%
def scaleAttentionWeights(input_id, attentionHeadsList):
    list_fwd_hooks = []

    for layer in range(32):
        for head in range(32):
            if((layer, head) in attentionHeadsList):
                list_fwd_hooks.append((utils.get_act_name("z", layer, "attn"), partial(scale_attention_heads, head_index=head, scale_number=(scaleFactor/(layer+1)))))
    scaled_logits = model.run_with_hooks(
            input_id, 
            fwd_hooks = list_fwd_hooks, 
            return_type="logits"
        )
    return scaled_logits

# %%
# scaled_logits = scaleAttentionWeights(input_id, bottom_k_heads)

# %%
# scaled_probability = torch.nn.functional.softmax(scaled_logits[:, -1, :], dim=1)[0, token_X_id].item()

# %%
# scaled_probability

# %%
def getProbability(logits, token_X_id):
    return torch.nn.functional.softmax(logits[:, -1, :], dim=1)[0, token_X_id].item()

# %%
# scaled_probability = getProbability(scaled_logits, token_X_id)
# probability = getProbability(logit, token_X_id)

# %%
# scaled_probability, probability

# %%
# if(type == 'top'):
    # attentionHeadsList = top_k_heads
# if(type == 'bottom'):
    # attentionHeadsList = bottom_k_heads
print(f"Number of common heads: {len(common_heads)}")

from tqdm import tqdm
average_normalized_probability = []
average_diff_probability = []
progressBar = tqdm(prompts, desc=f"Noise_Index {noise_index}")
proabability_list = []
with open('scalingResults.json', 'r') as f:
    results = json.load(f)
for prompt in progressBar:
    input_id = getInputId(prompt)
    logit, cache = getCacheLogits(input_id)
    token_X_id = torch.argmax(logit[:, -1, :], dim=1)[0]
    scaled_logits = scaleAttentionWeights(input_id, common_heads                                                                                                                                        )
    probability = getProbability(logit, token_X_id)
    scaled_probability = getProbability(scaled_logits, token_X_id)
    proabability_list.append({'probability': probability, 'scaled_probability': scaled_probability})
    
    
    
    normalized_probability = (scaled_probability - probability) / probability
    average_normalized_probability.append(normalized_probability)
    average_diff_probability.append(scaled_probability - probability)
    progressBar.set_description(f"average_normalized_probability: {sum(average_normalized_probability) / len(average_normalized_probability)}")
    
    
print(f"Average Normalized Probability: {sum(average_normalized_probability) / len(average_normalized_probability)}")
# with open(f'Noise_index_{noise_index}_scale_{scaleFactor}_bottom_k_heads_probability.json', 'w') as f:
    # json.dump(proabability_list, f)
results.append({'noise_index': noise_index, 'scale_factor': scaleFactor, 'number_of_examples': number_of_examples, 'common_heads': len(common_heads)
                , 'average_normalized_probability': sum(average_normalized_probability) / len(average_normalized_probability),
                'average_diff_probability': sum(average_diff_probability) / len(average_diff_probability)})

with open('scalingResults.json', 'w') as f:
    json.dump(results, f, indent=4)

# %%
scaleFactor

# %%



