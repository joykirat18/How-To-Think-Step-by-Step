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
# %%

import argparse
parser = argparse.ArgumentParser()
# python3 IndividualHeadAccuracy.py --noiseIndex 0 --device "cuda:1" --numExamples 10 --alteranteExamples 10
parser.add_argument("-noiseIndex", "--noiseIndex", help = "noiseIndex")
# parser.add_argument("-start", "--start", help = "start")
# parser.add_argument("-end", "--end", help = "end")
# parser.add_argument("-device", "--device", help = "device")
parser.add_argument("-examples", "--examples", help = "examples")


args = parser.parse_args()
# %%
import json
with open('/home/t-josingh/How-To-Think-Step-by-Step/data/activationPatching_llama2_clean.json', 'r') as f:
    COTData = json.load(f)

noise_index = int(args.noiseIndex)
number_of_examples = int(args.examples)

import sys
sys.path.append('../')
from utilsFile.fewShotConstants import TemplateCOT_fictional, TemplateCOT_false
# %%
print(f"Noise index: {noise_index}")
print(f"Number of examples: {number_of_examples}")
prompts = [TemplateCOT_fictional.format(data['prompt']) + data[f'response_{noise_index}'] for data in COTData][:number_of_examples]
labels = [(data[f'response_{noise_index + 1}'].replace(data[f'response_{noise_index}'], "").strip()) for data in COTData][:number_of_examples]

# %%
input_ids = []
max_length = 0
for prompt in prompts:
    finalPrompt = prompt
    encoded_prompt = tokenizer.encode(finalPrompt, add_special_tokens=False, return_tensors="pt")
    max_length = max(max_length, encoded_prompt.size()[1])
    input_ids.append(encoded_prompt[0])

# %%
pad_id = tokenizer.bos_token_id
def getInputId(prompt):
    encoded_prompt =tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    return encoded_prompt

def getCacheLogits(input_id):
    patched_cache = {}
    list_fwd_hooks = []
    def storeHookCache(value, hook):
        patched_cache[hook.name] = torch.from_numpy(value.detach().cpu().numpy())
    for layer in range(32):

        # list_fwd_hooks.append((utils.get_act_name("pattern", layer, "attn"), storeHookCache))
        # list_fwd_hooks.append((utils.get_act_name("resid_pre", layer), storeHookCache))
        list_fwd_hooks.append((utils.get_act_name("z", layer, "attn"), storeHookCache))
            
    patched_logits = model.run_with_hooks(
            input_id, 
            fwd_hooks = list_fwd_hooks, 
            return_type="logits"
        )
    return patched_logits, patched_cache
# %%
input_id = getInputId(prompts[0])
logit, cache = getCacheLogits(input_id)

# %%

umembed = model.unembed
ln_final = model.ln_final
def stage_I(pos, token_id, layer_number):
    from tqdm import tqdm
    prob_each_layer_head = []
    avg_prob = 0
    for layer in range(layer_number):
        for head in range(32):
            z: Float[Tensor, "batch seq d_head"] = cache[utils.get_act_name("z", layer, "attn")][:, :, head].to(device)
            N = z.size(0)
            output: Float[Tensor, "batch seq d_model"] = z @ model.W_O[layer, head]
            output = umembed(ln_final(output))
            output_end_token_prob = torch.softmax(output[:, pos, :], dim=-1)[0,token_id]
            prob_each_layer_head.append({'layer': layer, 'head': head, 'prob': output_end_token_prob.item()})
            avg_prob += output_end_token_prob.item()
    # avg_prob /= (32*layer_number)
    
    top_l = sorted(prob_each_layer_head, key=lambda x: x['prob'], reverse=True)
    top_l_layer_head = []
    for x in top_l:
        top_l_layer_head.append({'layer' : x['layer'], 'head' : x['head'], 'input_id' : token_id, 'pos' : pos, 'prob' : x['prob']})
    # top_l_layer_head = top_l_layer_head[:l]
    # print("top l layer heads")
    # print(top_l_layer_head)
    return top_l_layer_head

# %%
token_X_id = torch.argmax(logit[:, -1, :], dim=1)[0]
initial_layer_head = stage_I(-1, token_X_id, 32)

# %%
writing_heads = []
from tqdm import tqdm
for prompt in tqdm(prompts, desc=f"Noise_Index {noise_index}"):
    input_id = getInputId(prompt)
    logit, cache = getCacheLogits(input_id)
    token_X_id = torch.argmax(logit[:, -1, :], dim=1)[0]
    initial_layer_head = stage_I(-1, token_X_id, 32)
    writing_heads.append(initial_layer_head)
    with open(f'writing_heads_{noise_index}.pkl', 'wb') as f:
        pickle.dump(writing_heads, f)