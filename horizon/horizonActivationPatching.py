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
import argparse
parser = argparse.ArgumentParser()
# # python3 IndividualHeadAccuracy.py --noiseIndex 0 --device "cuda:1" --numExamples 10 --alteranteExamples 10
parser.add_argument("-noiseIndex", "--noiseIndex", help = "noiseIndex")
parser.add_argument("-device", "--device", help = "device")
# parser.add_argument("-numExamples", "--numExamples", help = "numExamples")
# parser.add_argument("-alternateExamples", "--alternateExamples", help = "alternateExamples")

# # alternateExamples
# parser.add_argument("--generate", action="store_true", help = "generate")
# # parser.add_argument("-modelPath", "--modelPath", help = "modelPath")
# parser.add_argument("--combined", action="store_true", help = "combined")
# # parser.add_argument("-combine_type", "--combine_type", help = "combine_type")

args = parser.parse_args()
# %%
torch.set_grad_enabled(False)
# %%
import json
with open('horizonDataset.json', 'r') as f:
    COTData = json.load(f)
# %%
import sys
sys.path.append('../')
from utilsFile.fewShotConstants import TemplateCOT_fictional, TemplateCOT_false
Template = TemplateCOT_fictional
# %%
# device = "cuda:2"
device = args.device if torch.cuda.is_available() else "cpu"
noise_index = int(args.noiseIndex)
# noise_index = 1
# %%
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from utilsFile.loadModel import loadTransformerLensModel, runCacheActivationPatching
MODEL_PATH = '/home/models//Llama-2-7b-hf'
# modelPath='/home/models/vicuna-7b'
model, tokenizer = loadTransformerLensModel(MODEL_PATH)
model = model.to(device)
# %%
prompts = [data['prompt'] for data in COTData]
noisePrompts = [data['noise_prompt'] for data in COTData]
responses = [data[f'response_{noise_index}'] for data in COTData]
answers = [data['answer'] for data in COTData]
# %%
def patch_layer_vector(
    clean_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
    hook, 
    noise_cache):
    # print(clean_head_vector)
    for i in range(32):
        clean_head_vector[:, :, i] = noise_cache[hook.name][:, :, i]
    return clean_head_vector
# %%
def generate_text(input_id, noise_input_id,layer, max_new_tokens):
        original_input_id = input_id.clone()
        while (max_new_tokens > 0):
            original_logit, cache = runCacheActivationPatching(model, input_id)
            noise_logit, noise_cache = runCacheActivationPatching(model, noise_input_id) 
            hook_fn = partial(patch_layer_vector, noise_cache=noise_cache)        
            patched_logit = model.run_with_hooks(
                    input_id,
                    fwd_hooks = [(utils.get_act_name("z", layer, "attn"), 
                    hook_fn)],
                    return_type="logits"
                )
            next_token_id = torch.argmax(patched_logit[:, -1, :], dim=-1)
            next_token = model.tokenizer.convert_ids_to_tokens(next_token_id)[0]
            print(next_token)
            if(next_token == '<0x0A>'):
                break
            input_id = torch.cat([input_id, next_token_id.unsqueeze(1)], dim=-1)
            noise_input_id = torch.cat([noise_input_id, next_token_id.unsqueeze(1)], dim=-1)
            max_new_tokens -= 1
        generated_text = model.tokenizer.decode(input_id[0][original_input_id.size(1):], skip_special_tokens=True)
        return generated_text
# %%
save_dir = f"HorizonResults/{noise_index}"
import os
os.makedirs(save_dir, exist_ok=True)

# %%
result = []
pad_id = tokenizer.bos_token_id
from tqdm import tqdm
for i in tqdm(range(50)):
    result.append({})
    prompt = TemplateCOT_fictional.format(prompts[i]) + responses[i]
    noise_prompt = TemplateCOT_fictional.format(noisePrompts[i]) + responses[i]
    input_id = tokenizer(prompt, return_tensors="pt").input_ids
    noise_input_id = tokenizer(noise_prompt, return_tensors="pt").input_ids

    max_seq_len = max(len(input_id[0]), len(noise_input_id[0]))
    input_id = [pad_id] * (max_seq_len - len(input_id[0])) + input_id[0].detach().cpu().numpy().tolist()
    noise_input_id = [pad_id] * (max_seq_len - len(noise_input_id[0])) + noise_input_id[0].detach().cpu().numpy().tolist()
    input_id = torch.tensor(input_id).unsqueeze(0).to(device)
    noise_input_id = torch.tensor(noise_input_id).unsqueeze(0).to(device)

    result[i]['prompt'] = prompts[i]
    result[i]['noise_prompt'] = noisePrompts[i]
    result[i]['answer'] = answers[i]
    for layer in tqdm(range(32), desc=f'layer : '):
        generated_text = generate_text(input_id, noise_input_id, layer, 50)
        result[i][f'layer_{layer}'] = generated_text
        print(generated_text)
        with open(f'{save_dir}/horizonLLAMA2({noise_index}).json', 'w') as f:
            json.dump(result, f)        

# %%
