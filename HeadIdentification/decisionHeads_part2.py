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
 
# # Adding optional argument
parser.add_argument("-noiseIndex", "--noiseIndex", help = "noiseIndex")
parser.add_argument("-device", "--device", help = "device")
parser.add_argument("-numExamples", "--numExamples", help = "numExamples")
# parser.add_argument("--continueTraining", action="store_true", help = "continueTraining")
parser.add_argument("-modelPath", "--modelPath", help = "modelPath")

args = parser.parse_args()


def imshow(tensor, renderer=None, save='head.html', **kwargs):
    fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs)
    # fig.show(renderer)
    fig.write_html(save)


# %%
device = args.device if torch.cuda.is_available() else "cpu"
# device = 'cuda:1' if torch.cuda.is_available() else "cpu"

# device = "cpu"

# %%
import json
with open('data/activationPatching_2hop_with_index_2.json', 'r') as f:
    COTData = json.load(f)
# %%
from transformers import LlamaForCausalLM, LlamaTokenizer
import os

# MODEL_PATH = args.modelPath
MODEL_PATH='/home/models/vicuna-7b'
from utilsFile.loadModel import loadTransformerLensModel
model, tokenizer = loadTransformerLensModel(MODEL_PATH)
model = model.to(device)

# %%


# %%
from utilsFile.fewShotConstants import TemplateCOT_fictional
Template = TemplateCOT_fictional
print(Template)
# %%
noise_index = int(args.noiseIndex)
number_of_examples = int(args.numExamples)


print(f"Noise index: {noise_index}")
print(f"Number of examples: {number_of_examples}")

import os
save_dir = f"results/decision/part_2/{noise_index}"
os.makedirs(save_dir, exist_ok=True)
prompts = [Template.format(data['prompt']) + data[f'response_{noise_index}'] for data in COTData]

word_pos_end = [data[f'end_token_pos_{noise_index}'] for data in COTData][:number_of_examples]
word_pos_answer = [data[f'answer_token_pos_{noise_index}'] for data in COTData][:number_of_examples]

word_id_answer = [data[f'answer_token_id_{noise_index}'] for data in COTData][:number_of_examples]
word_id_end = [data[f'end_token_id_{noise_index}'] for data in COTData][:number_of_examples]


# %%
input_ids = []
max_length = 0
for prompt in prompts:
    finalPrompt = prompt
    encoded_prompt = tokenizer.encode(finalPrompt, return_tensors="pt")
    max_length = max(max_length, encoded_prompt.size()[1])
    input_ids.append(encoded_prompt[0])

# %%
max_seq_len = max(max_length, max_length)

# %%
pad_id = tokenizer.bos_token_id
input_ids = input_ids[:number_of_examples]

# %%
# padd all input_ids to max length from left
for i in range(len(input_ids)):
    length_input_id = len(input_ids[i])
    input_ids[i] = [pad_id] * (max_seq_len - length_input_id) + input_ids[i].detach().numpy().tolist()

input_ids = torch.tensor(input_ids)

# %%
pad_id = tokenizer.bos_token_id

# %%
print("RUNNING ORIGINAL LOGITS")
fwd_hooks_list = []
cache = {}
def storeHookCache(value, hook):
    cache[hook.name] = torch.from_numpy(value.detach().cpu().numpy())
for layer in range(model.cfg.n_layers):
    fwd_hooks_list.append((utils.get_act_name("resid_mid", layer), storeHookCache))
    fwd_hooks_list.append((utils.get_act_name("resid_pre", layer), storeHookCache))
    fwd_hooks_list.append((utils.get_act_name("z", layer), storeHookCache))
# %%
model.run_with_hooks(input_ids, return_type=None, fwd_hooks=fwd_hooks_list)
# %%
prob_mid = torch.zeros(model.cfg.n_layers, device=device, dtype=torch.float32)
# %%
unembed = model.unembed
ln_final = model.ln_final
# %%
from tqdm import tqdm
for i in tqdm(range(32)):
    resid_mid = cache[utils.get_act_name("resid_mid", i)].to(device)

    resid_mid_logit = unembed(ln_final(resid_mid))
    resid_mid_logit = torch.softmax(resid_mid_logit[:, -1, :], dim=-1)
    for j in range(len(word_id_answer)):
        prob_mid[i] += resid_mid_logit[j, word_id_answer[j]]

    prob_mid[i] = prob_mid[i] / len(word_id_answer)


# %%
prob_mid_value = []
label = []
for i in range(32):
    prob_mid_value.append(prob_mid[i].item())
    label.append(utils.get_act_name("resid_mid", i))
# %%
individual_head_prob = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
import copy
from tqdm import tqdm
for layer in tqdm(range(model.cfg.n_layers)):
    for head in tqdm(range(model.cfg.n_heads)):
        z = copy.deepcopy(cache[utils.get_act_name("z", layer)])
        head_value = copy.deepcopy(z[:, :, head, :])
        z[:, :, :, :] = 0
        z[:, :, head, :] = head_value

        residual_pre = cache[utils.get_act_name("resid_pre", layer)]
        residual_mid = cache[utils.get_act_name("resid_mid", layer)]
        attn_out_cal = einops.einsum(
                    z, model.blocks[layer].attn.W_O.detach().cpu(),
                    "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
                ) + model.blocks[layer].attn.b_O.detach().cpu()
        residual_mid_cal = attn_out_cal + residual_pre
        prob = 0
        resid_mid_logit = unembed(ln_final(residual_mid_cal.to(device)))
        resid_mid_logit = torch.softmax(resid_mid_logit[:, -1, :], dim=-1)
        for j in range(len(word_id_answer)):
            prob += resid_mid_logit[j, word_id_answer[j]]
        prob = prob / len(word_id_answer)
        # print(prob_mid[layer], prob)
        individual_head_prob[layer, head] = prob
# %%
imshow(individual_head_prob,save=f'{save_dir}/raw_prob.html', title="Residual mid probability with patched heads", labels={"x":"Head", "y":"Layer"}) 

# %%
normalised_prob = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        normalised_prob[layer, head] = (prob_mid[layer] - individual_head_prob[layer, head]) / prob_mid[layer]
# %%
imshow(normalised_prob,save=f'{save_dir}/normalised_prob.html', title="normalised Residual mid probability with patched heads", labels={"x":"Head", "y":"Layer"}) 
# %%
import pickle
with open(f'{save_dir}/normalised_prob.pkl', 'wb') as f:
    pickle.dump(normalised_prob, f)
with open(f'{save_dir}/raw_head_prob.pkl', 'wb') as f:
    pickle.dump(individual_head_prob, f)
with open(f'{save_dir}/prob_mid.pkl', 'wb') as f:
    pickle.dump(prob_mid, f)

# %%
import plotly.express as px
import pandas as pd
def line(tensor, label, renderer=None, xaxis="", yaxis="", hover_labels=None, **kwargs):
    # Create a DataFrame with x and y data
    df = pd.DataFrame({'x': label, 'y': tensor})
    
    # Create a figure using plotly express
    fig = px.line(df, x='x', y='y', labels={"x": xaxis, "y": yaxis}, hover_data=hover_labels, **kwargs)
 
    fig.write_html(f'{save_dir}/residualProb_mid.html')
# %%
line(prob_mid_value, label, yaxis="Probability", xaxis="Layer")
# %%