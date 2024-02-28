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
# # Initialize parser
import argparse
parser = argparse.ArgumentParser()
 
parser.add_argument("-noiseIndex", "--noiseIndex", help = "noiseIndex")
parser.add_argument("-device", "--device", help = "device")
parser.add_argument("-numExamples", "--numExamples", help = "numExamples")
parser.add_argument("--continueTraining", action="store_true", help = "continueTraining")
parser.add_argument("-modelPath", "--modelPath", help = "modelPath")

args = parser.parse_args()

# %%
def imshow(tensor, renderer=None, save='head.html', **kwargs):
    fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs)
    # fig.show(renderer)
    fig.write_html(save)

# %%
device = args.device if torch.cuda.is_available() else "cpu"
# device = "cuda:2" if torch.cuda.is_available() else "cpu"

# %%

from transformers import LlamaForCausalLM, LlamaTokenizer
import os

MODEL_PATH = args.modelPath
# MODEL_PATH='/home/models/vicuna-7b'
print(MODEL_PATH)
from utilsFile.loadModel import loadTransformerLensModel
model, tokenizer = loadTransformerLensModel(MODEL_PATH)
model = model.to(device)

# %%
import json
with open('data/simple_1hop.json', 'r') as f:
    original = json.load(f)

# %%
index_to_replace = int(args.noiseIndex)
number_of_examples = int(args.numExamples)
continueTraining = args.continueTraining


import os
save_dir = f"results/reasoning/{index_to_replace}"
os.makedirs(save_dir, exist_ok=True)

prompts = [data['input'] for data in original]
label = [data['label'] for data in original]
noise_prompts = [data[f'noise_input_{index_to_replace}'] for data in original]
noise_label = [data[f'noise_label_{index_to_replace}'] for data in original]



# %%

# Template = """Complete the following sentence: {}"""
# Template = """### Instruction:
# Answe"""
# %%
from utilsFile.fewShotConstants import Template_simple_1hop
Template = Template_simple_1hop
print(Template)
# %%
input_ids = []
max_length = 0
for prompt in prompts:
    finalPrompt = Template.format(prompt)
    # print(finalPrompt)
    encoded_prompt = tokenizer.encode(finalPrompt, add_special_tokens=False, return_tensors="pt")
    max_length = max(max_length, encoded_prompt.size()[1])
    input_ids.append(encoded_prompt[0])


# %%
noise_max_length = 0
noise_input_ids = []
for prompt in noise_prompts:
    # print(prompt)
    finalPrompt = Template.format(prompt)
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
def model_accuracy(model, input_ids, labels):
    model_accuracy = 0
    N = len(input_ids)
    for i in range(N):
        id = input_ids[i]
        with torch.no_grad():
            outputs = model(id)
            next_token_logits = outputs[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        next_token = model.tokenizer.decode(next_token_id).strip()
        # print(next_token, labels[i])
        if(labels[i].strip().startswith(next_token.strip())):
            model_accuracy += 1
        # if len(next_token) != 1 and next_token in dataset[i]['label']:
            # model_accuracy += 1
    return model_accuracy/N

# %%
def logit_accuracy(logits, labels):
    next_token_logits = logits[:, -1, :]
    next_token_ids = torch.argmax(next_token_logits, dim=-1)
    next_token_ids = [[tokens.item()] for tokens in next_token_ids]
    accuracy = 0
    next_tokens = model.tokenizer.batch_decode(next_token_ids)
    # print(next_tokens)
    for i in range(len(next_tokens)):
        if(labels[i].strip().startswith(next_tokens[i].strip())):
            accuracy += 1
    return accuracy / len(labels)

def get_end_idxs(toks, tokenizer, name_tok_len=1, prepend_bos=False):
    # toks = torch.Tensor(tokenizer([prompt["input"] for prompt in prompts], padding=True).input_ids).type(torch.int)
    relevant_idx = int(prepend_bos)
    # if the sentence begins with an end token
    # AND the model pads at the end with the same end token,
    # then we need make special arrangements

    pad_token_id = tokenizer.pad_token_id

    end_idxs_raw = []
    for i in range(toks.shape[0]):
        if pad_token_id not in toks[i][1:]:
            end_idxs_raw.append(toks.shape[1])
            continue
        nonzers = (toks[i] == pad_token_id).nonzero()
        try:
            nonzers = nonzers[relevant_idx]
        except:
            print(toks[i])
            print(nonzers)
            print(relevant_idx)
            print(i)
            raise ValueError("Something went wrong")
        nonzers = nonzers[0]
        nonzers = nonzers.item()
        end_idxs_raw.append(nonzers)
    end_idxs = torch.tensor(end_idxs_raw)
    end_idxs = end_idxs - 1 - name_tok_len

    for i in range(toks.shape[0]):
        assert toks[i][end_idxs[i] + 1] != 0 and (
            toks.shape[1] == end_idxs[i] + 2 or toks[i][end_idxs[i] + 2] == pad_token_id
        ), (
            toks[i],
            end_idxs[i],
            toks[i].shape,
            "the END idxs aren't properly formatted",
        )

    return end_idxs

def logit_diff(logits, input_ids, label, noise_label):
    label_idx = []
    for l in label:
        label_idx.append(tokenizer(l).input_ids[1])
    noise_label_idx = []
    for l in noise_label:
        noise_label_idx.append(tokenizer(l).input_ids[1])
    end_idx = get_end_idxs(input_ids, tokenizer)
    label_logits = logits[torch.arange(len(input_ids)), end_idx, label_idx]
    noise_label_logits = logits[torch.arange(len(input_ids)), end_idx, noise_label_idx]
    return (label_logits - noise_label_logits).mean().detach().cpu()
# %%
filtered_label = []
filtered_noise_label = []
filtered_input_ids = []
filtered_noise_input_ids = []
from tqdm import tqdm
for i in tqdm(range(len(input_ids))):
    # print(i, len(filtered_label))
    if(len(filtered_label) == number_of_examples):
        break
    originalCorrect = model_accuracy(model, input_ids[i].unsqueeze(0), [label[i]])
    noiseCorrectWithNoiseLabel = model_accuracy(model, noise_input_ids[i].unsqueeze(0), [noise_label[i]])

    if(originalCorrect == 1 and noiseCorrectWithNoiseLabel == 1):
        print(len(filtered_label))

        filtered_label.append(label[i])
        filtered_noise_label.append(noise_label[i])
        filtered_input_ids.append(input_ids[i].detach().cpu().numpy())
        filtered_noise_input_ids.append(noise_input_ids[i].detach().cpu().numpy())
# %%

filtered_input_ids = torch.tensor(np.array(filtered_input_ids))
filtered_noise_input_ids = torch.tensor(np.array(filtered_noise_input_ids))
# # %%

label = filtered_label
noise_label = filtered_noise_label
input_ids = filtered_input_ids
noise_input_ids = filtered_noise_input_ids

print("Length of input_ids: ", len(input_ids))
# %%
print("RUNNING ORIGINAL LOGITS")

# %%
model.reset_hooks()
fwd_hooks_list = []
cache = {}
def storeHookCache(value, hook):
    cache[hook.name] = torch.from_numpy(value.detach().cpu().numpy())
for layer in range(model.cfg.n_layers):
    fwd_hooks_list.append((utils.get_act_name("z", layer, "attn"), storeHookCache))

original_logits = model.run_with_hooks(input_ids, return_type="logits", fwd_hooks=fwd_hooks_list)

# original_logits, cache = model.run_with_cache(input_ids, device='cpu')

# %%
print("RUNNING NOISE LOGITS")
model.reset_hooks()
fwd_hooks_list = []
noise_cache = {}
def storeHookNoiseCache(value, hook):
    noise_cache[hook.name] = torch.from_numpy(value.detach().cpu().numpy())
for layer in range(model.cfg.n_layers):
    fwd_hooks_list.append((utils.get_act_name("z", layer, "attn"), storeHookNoiseCache))
    
noise_logits = model.run_with_hooks(noise_input_ids, return_type="logits", fwd_hooks=fwd_hooks_list)
# noise_logits, noise_cache = model.run_with_cache(noise_input_ids, device='cpu')

# %%
# print(label)
from utilsFile.evaluation import kl_divergence, kl_divergence_mean_var, wasserstein_dis
# print(noise_label)
original_acc = logit_accuracy(original_logits, label)
noise_acc = logit_accuracy(noise_logits, label)
noise_acc_with_noise_label = logit_accuracy(noise_logits, noise_label)

print(f"Original Accuracy: {original_acc}")
print(f"Noise Accuracy: {noise_acc}")
print(f"Noise Accuracy with noise label: {noise_acc_with_noise_label}")

original_logit_diff = logit_diff(original_logits, input_ids, label, noise_label)
noise_logit_diff = logit_diff(noise_logits, noise_input_ids, label, noise_label)

print(f"Original Logit Diff: {original_logit_diff}")
print(f"Noise Logit Diff: {noise_logit_diff}")
# breakpoint()
noise_wasserstein_distance = wasserstein_dis(noise_logits, original_logits)
print(f"Noise Wasserstein Distance: {noise_wasserstein_distance}")
# %%

# %%
noise_kl_divergence = kl_divergence(noise_logits, original_logits)
print("Noise kl_divergence: ", noise_kl_divergence)
# %%
from tqdm import tqdm

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
def normalize_patched_kl_div(patched_kl_div):
    # Subtract corrupted logit diff to measure the improvement, divide by the total improvement from clean to corrupted to normalise
    # 0 means zero change, negative means actively made worse, 1 means totally recovered clean performance, >1 means actively *improved* on clean performance
    return (noise_kl_divergence - patched_kl_div) / (noise_kl_divergence)

def normalize_patched_logit_diff(patched_logit_diff):
    # Subtract corrupted logit diff to measure the improvement, divide by the total improvement from clean to corrupted to normalise
    # 0 means zero change, negative means actively made worse, 1 means totally recovered clean performance, >1 means actively *improved* on clean performance
    return (noise_logit_diff - patched_logit_diff) / (noise_logit_diff)

def normalize_patched_wasserstein_distance(patched_wasserstein_distance):
    # Subtract corrupted logit diff to measure the improvement, divide by the total improvement from clean to corrupted to normalise
    # 0 means zero change, negative means actively made worse, 1 means totally recovered clean performance, >1 means actively *improved* on clean performance
    return (noise_wasserstein_distance - patched_wasserstein_distance) / (noise_wasserstein_distance)

patched_head_kl_divergence = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
patched_head_logit_diff = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
patched_head_kl_divergence_mean_var = np.zeros((model.cfg.n_layers, model.cfg.n_heads), dtype=object)
patched_head_wasserstein_distance = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)

# patched_head_diff_noise_label = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device="cuda", dtype=torch.float32)

# %%
for layer in tqdm(range(model.cfg.n_layers)):
    # count += 1
    # print(count)
    for head_index in tqdm(range(model.cfg.n_heads)):
        # print(patched_head_z_diff[layer][head_index])
        # breakpoint()

        hook_fn = partial(patch_head_vector, head_index=head_index, clean_cache=cache)
        patched_logits = model.run_with_hooks(
            noise_input_ids, 
            fwd_hooks = [(utils.get_act_name("z", layer, "attn"), 
                hook_fn)], 
            return_type="logits"
        )
        patched_wasserstein_distance = wasserstein_dis(patched_logits, original_logits)
        patched_kl_div = kl_divergence(patched_logits, original_logits)
        patched_logit_diff = logit_diff(patched_logits, noise_input_ids, label, noise_label)
        

        patched_head_kl_divergence_mean_var = kl_divergence_mean_var(patched_logits, original_logits)


        patched_head_kl_divergence[layer, head_index] = normalize_patched_kl_div(patched_kl_div)
        patched_head_logit_diff[layer, head_index] = normalize_patched_logit_diff(patched_logit_diff)
        patched_head_wasserstein_distance[layer, head_index] = normalize_patched_wasserstein_distance(patched_wasserstein_distance)
        # print(patched_head_z_diff[layer, head_index])

        import pickle
        with open(f'{save_dir}/kl_div.pickle', 'wb') as f:
            pickle.dump(patched_head_kl_divergence, f)
        
        import pickle
        with open(f'{save_dir}/logit_diff.pickle', 'wb') as f:
            pickle.dump(patched_head_logit_diff, f)
        
        import pickle
        with open(f'{save_dir}/kl_div_mean_var.pickle', 'wb') as f:
            pickle.dump(patched_head_kl_divergence_mean_var, f)
        
        import pickle
        with open(f'{save_dir}/wasserstein_distance.pickle', 'wb') as f:
            pickle.dump(patched_head_wasserstein_distance, f)
# # %%
imshow(patched_head_kl_divergence,save=f'{save_dir}/head_kl_div_reasoning.html', title="Kl Divergence with label From Patched Head Output reasoning", labels={"x":"Head", "y":"Layer"})
imshow(patched_head_logit_diff,save=f'{save_dir}/logit_diff_reasoning.html', title="Logit Difference with label From Patched Head Output reasoning", labels={"x":"Head", "y":"Layer"})
imshow(patched_head_wasserstein_distance,save=f'{save_dir}/wasserstein_distance_reasoning.html', title="Wasserstein Distance with label From Patched Head Output reasoning", labels={"x":"Head", "y":"Layer"})

# %%
