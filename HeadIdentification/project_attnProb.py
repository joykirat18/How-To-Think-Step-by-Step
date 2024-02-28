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

# %%
# def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
#     px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

# def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
#     px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

# def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
#     x = utils.to_numpy(x)
#     y = utils.to_numpy(y)
#     px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

# %%
# def imshow(tensor, renderer=None, **kwargs):
#     px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs).show(renderer)

# def imshow(tensor, renderer=None, save='head.html', **kwargs):
#     fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs)
#     # fig.show(renderer)
#     fig.write_html(save)

# def line(tensor, renderer=None, **kwargs):
#     px.line(y=utils.to_numpy(tensor), **kwargs).show(renderer)

# def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
#     x = utils.to_numpy(x)
#     y = utils.to_numpy(y)
#     px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

# %%
device = args.device if torch.cuda.is_available() else "cpu"
# device = 'cuda:2' if torch.cuda.is_available() else "cpu"

# device = "cpu"
import json
with open('data/activationPatching_llama2_clean.json', 'r') as f:
    COTData = json.load(f)
# %%
from transformers import LlamaForCausalLM, LlamaTokenizer
import os

# MODEL_PATH = args.modelPath
MODEL_PATH='/home/models/Llama-2-7b-hf'
from utilsFile.loadModel import loadTransformerLensModel
model, tokenizer = loadTransformerLensModel(MODEL_PATH)
model = model.to(device)

# %%



# %%

# import sys
# sys.path.append('../')
from utilsFile.fewShotConstants import TemplateCOT_fictional
Template = TemplateCOT_fictional
print(Template)
# %%
noise_index = int(args.noiseIndex)
number_of_examples = int(args.numExamples)
# continueTraining = args.continueTraining

# noise_index = 0
# number_of_examples = 5
# continueTraining = False

print(f"Noise index: {noise_index}")
print(f"Number of examples: {number_of_examples}")

# prompts = []
# word_pos_end = []
# word_pos_answer = []
# word_id_answer = []
# word_id_end = []
# count = 0
# while(count <= number_of_examples):
#     if(f'answer_token_id_{noise_index}' not in COTData[count].keys()):
#         count += 1
#         continue
#     prompts.append(Template.format(COTData[count]['prompt']) + COTData[count][f'response_{noise_index}'])
#     word_pos_end.append(COTData[count][f'end_token_pos_{noise_index}'])
#     word_pos_answer.append(COTData[count][f'answer_token_pos_{noise_index}'])
#     word_id_answer.append(COTData[count][f'answer_token_id_{noise_index}'])
#     word_id_end.append(COTData[count][f'end_token_id_{noise_index}'])
#     count += 1
import os
os.makedirs(f'results/copy/{noise_index}', exist_ok=True)
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
print(max_seq_len)
# %%
pad_id = tokenizer.bos_token_id
input_ids = input_ids[:number_of_examples]
# %%
# padd all input_ids to max length from left
for i in range(len(input_ids)):
    length_input_id = len(input_ids[i])
    # print(length_input_id, (max_seq_len - length_input_id), word_pos_end[i])
    input_ids[i] = [pad_id] * (max_seq_len - length_input_id) + input_ids[i].detach().numpy().tolist()
    # word_pos_answer[i] = word_pos_answer[i] + (max_seq_len - length_input_id)
    # word_pos_end[i] = word_pos_end[i] + (max_seq_len - length_input_id)
    # print(word_pos_end[i])
input_ids = torch.tensor(input_ids)

# %%
pad_id = tokenizer.bos_token_id

# %%
from utilsFile.loadModel import runCacheCopying
print("RUNNING ORIGINAL LOGITS")
# breakpoint()
def runCacheCopying(model, input_ids):
    model.reset_hooks()
    fwd_hooks_list = []
    cache = {}
    def storeHookCache(value, hook):
        cache[hook.name] = torch.from_numpy(value.detach().cpu().numpy())
    for layer in range(model.cfg.n_layers):
        fwd_hooks_list.append((utils.get_act_name("z", layer, "attn"), storeHookCache))
        fwd_hooks_list.append((utils.get_act_name("pattern", layer, "attn"), storeHookCache))

    model.run_with_hooks(input_ids, return_type=None, fwd_hooks=fwd_hooks_list)

    return cache
cache = runCacheCopying(model, input_ids)
# %%
from torch import Tensor
def calculate_and_show_scatter_embedding_vs_attn(
    layer: int,
    head: int,
    cache: ActivationCache,
) -> None:
    '''
    Creates and plots a figure equivalent to 3(c) in the paper.

    This should involve computing the four 1D tensors:
        attn_from_end_to_io
        attn_from_end_to_s
        projection_in_io_dir
        projection_in_s_dir
    and then calling the scatter_embedding_vs_attn function.
    '''
    # SOLUTION
    # Get the value written to the residual stream at the end token by this head
    z: Float[Tensor, "batch seq d_head"] = cache[utils.get_act_name("z", layer)][:, :, head].to(device)
    N = z.size(0)
    output: Float[Tensor, "batch seq d_model"] = z @ model.W_O[layer, head]
    output_on_end_token: Float[Tensor, "batch d_model"] = output[torch.arange(N), word_pos_end]

    # Get the directions we'll be projecting onto
    answer_unembedding: Float[Tensor, "batch d_model"] = model.W_U.T[word_id_answer]

    # Get the value of projections, by multiplying and summing over the d_model dimension
    projection_in_answer_dir: Float[Tensor, "batch"] = (output_on_end_token * answer_unembedding).sum(-1)

    # Get attention probs, and index to get the probabilities from END -> IO / S
    attn_probs: Float[Tensor, "batch q k"] = cache[utils.get_act_name("pattern", layer)][:, head]
    attn_from_end_to_answer = attn_probs[torch.arange(N), word_pos_end, word_pos_answer]

    del z, output, output_on_end_token, answer_unembedding, attn_probs

    return projection_in_answer_dir, attn_from_end_to_answer


# %%
projection = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
attn_prob = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
from tqdm import tqdm
for layer in tqdm(range(model.cfg.n_layers)):
    for head in (range(model.cfg.n_heads)):
        projection_in_answer_dir, attn_from_end_to_answer = calculate_and_show_scatter_embedding_vs_attn(layer, head, cache)

        projection[layer, head] = projection_in_answer_dir.mean()
        attn_prob[layer, head] = attn_from_end_to_answer.mean()
# %%
import pickle
with open(f'results/copy/{noise_index}/projection.pkl', 'wb') as f:
    pickle.dump(projection, f)
with open(f'results/copy/{noise_index}/attn_prob.pkl', 'wb') as f:
    pickle.dump(attn_prob, f)
from utilsFile.showPlots import imshow
imshow(projection,save=f'results/copy/{noise_index}/projection.html', title=f"Projection in the answer direction for index {noise_index}", labels={"x":"Head", "y":"Layer"})
imshow(attn_prob, save=f'results/copy/{noise_index}/attn_prob.html', title=f"Attention probability on answer index {noise_index}", labels={"x":"Head", "y":"Layer"})

# %%
