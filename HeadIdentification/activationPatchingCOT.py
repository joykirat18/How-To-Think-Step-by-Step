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
from utilsFile.evaluation import kl_divergence

# %%
import circuitsvis as cv
# Testing that the library works
cv.examples.hello("Neel")

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
# Initialize parser
import argparse
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-noiseIndex", "--noiseIndex", help = "noiseIndex")
parser.add_argument("-device", "--device", help = "device")
parser.add_argument("-numExamples", "--numExamples", help = "numExamples")
parser.add_argument("--continueTraining", action="store_true", help = "continueTraining")
parser.add_argument("-modelPath", "--modelPath", help = "modelPath")

args = parser.parse_args()

# %%
device = args.device if torch.cuda.is_available() else "cpu"
# device = "cpu"

# %%
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from utilsFile.loadModel import loadTransformerLensModel, runCacheActivationPatching
MODEL_PATH = args.modelPath

# %%
model, tokenizer = loadTransformerLensModel(MODEL_PATH)
model = model.to(device)

# %%
import json
with open('data/activationPatching_2hop.json', 'r') as f:
    COTData = json.load(f)
# %%

# %%

from utilsFile.fewShotConstants import TemplateCOT_fictional
from utilsFile.showPlots import imshow
Template = TemplateCOT_fictional
print(Template)
# %%
noise_index = int(args.noiseIndex)
number_of_examples = int(args.numExamples)
continueTraining = args.continueTraining
import os
save_dir = f"results/activationPatching/{noise_index}"
os.makedirs(save_dir, exist_ok=True)

print(f"Noise index: {noise_index}")
print(f"Number of examples: {number_of_examples}")
prompts = [Template.format(data['prompt']) + data[f'response_{noise_index}'] for data in COTData][:number_of_examples]

noise_prompts = [Template.format(data[f'noise_prompt_{noise_index}']) + data[f'noise_response_{noise_index}'] for data in COTData][:number_of_examples]

# %%
input_ids = []
max_length = 0
for prompt in prompts:
    finalPrompt = prompt
    encoded_prompt = tokenizer.encode(finalPrompt, add_special_tokens=False, return_tensors="pt")
    max_length = max(max_length, encoded_prompt.size()[1])
    input_ids.append(encoded_prompt[0])

# %%
noise_max_length = 0
noise_input_ids = []
for prompt in noise_prompts:
    # print(prompt)
    finalPrompt = prompt
    # print(model.generate(finalPrompt))
    encoded_prompt = tokenizer.encode(finalPrompt, add_special_tokens=False, return_tensors="pt")
    noise_max_length = max(noise_max_length, encoded_prompt.size()[1])
    noise_input_ids.append(encoded_prompt[0])
# %%
max_seq_len = max(max_length, noise_max_length)

# %%
pad_id = tokenizer.bos_token_id

# %%
# padd all input_ids to max length from left
for i in range(len(input_ids)):
    input_ids[i] = [pad_id] * (max_seq_len - len(input_ids[i])) + input_ids[i].detach().numpy().tolist()
input_ids = torch.tensor(input_ids)

# %%
for i in range(len(noise_input_ids)):
    noise_input_ids[i] = [pad_id] * (max_seq_len - len(noise_input_ids[i])) + noise_input_ids[i].detach().numpy().tolist()
noise_input_ids = torch.tensor(noise_input_ids)
# %%
print("RUNNING ORIGINAL LOGITS")
original_logits, cache = runCacheActivationPatching(model, input_ids)
# original_logits, cache = model.run_with_cache(input_ids, device='cpu')

# %%
print("RUNNING NOISE LOGITS")
noise_logits, noise_cache = runCacheActivationPatching(model, noise_input_ids)
# %%

# %%
noise_kl_divergence = kl_divergence(noise_logits, original_logits)
print("Noise kl_divergence: ", noise_kl_divergence)
    # %%
from tqdm import tqdm
# %%
# input_ids.shape

# %%
def patch_head_vector(
    corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
    hook, 
    head_index, 
    clean_cache):
    # print(hook.name, head_index, corrupted_head_vector.shape)
    corrupted_head_vector[:, :, head_index, :] = clean_cache[hook.name][:, :, head_index, :]
    return corrupted_head_vector

count = 0
def normalize_patched_logit_diff(patched_logit_diff):
    # Subtract corrupted logit diff to measure the improvement, divide by the total improvement from clean to corrupted to normalise
    # 0 means zero change, negative means actively made worse, 1 means totally recovered clean performance, >1 means actively *improved* on clean performance
    return (noise_kl_divergence - patched_logit_diff) / (noise_kl_divergence)

patched_head_z_diff = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
import pickle

if(continueTraining):
    with open(f'{save_dir}/kl_div_COT.pickle', 'rb') as f:
        patched_head_z_diff = pickle.load(f)            
# patched_head_diff_noise_label = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device="cuda", dtype=torch.float32)

for layer in tqdm(range(model.cfg.n_layers)):
    # count += 1
    # print(count)
    for head_index in tqdm(range(model.cfg.n_heads)):
        # print(patched_head_z_diff[layer][head_index])
        # breakpoint()
        if(patched_head_z_diff[layer][head_index].item() != 0):
            continue

        hook_fn = partial(patch_head_vector, head_index=head_index, clean_cache=cache)
        patched_logits = model.run_with_hooks(
            noise_input_ids, 
            fwd_hooks = [(utils.get_act_name("z", layer, "attn"), 
                hook_fn)], 
            return_type="logits"
        )
        patched_logit_diff = kl_divergence(patched_logits, original_logits)
        # print(patched_logit_diff)

        patched_head_z_diff[layer, head_index] = normalize_patched_logit_diff(patched_logit_diff)
        # print(patched_head_z_diff[layer, head_index])

        import pickle
        with open(f'{save_dir}/kl_div_COT.pickle', 'wb') as f:
            pickle.dump(patched_head_z_diff, f)
# # %%
imshow(patched_head_z_diff,save=f'{save_dir}/kl_div_COT.html', title=f"KL Div Difference with noise index {noise_index} From Patched Head Output", labels={"x":"Head", "y":"Layer"})

