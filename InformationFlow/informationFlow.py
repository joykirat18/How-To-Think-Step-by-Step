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

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
# %%
torch.set_grad_enabled(False)
# %%
reasoning_heads = [{'noise_index': 0, 'accuracy_index': 2, 'accuracy': 1.0, 'head_removed': 475},
 {'noise_index': 1,  'accuracy_index': 3,  'accuracy': 0.93,  'head_removed': 554},
 {'noise_index': 2,  'accuracy_index': 4,  'accuracy': 0.96,  'head_removed': 617},
 {'noise_index': 3,  'accuracy_index': 4,  'accuracy': 0.97,  'head_removed': 617},
 {'noise_index': 4,  'accuracy_index': 3,  'accuracy': 0.99,  'head_removed': 554},
 {'noise_index': 5,  'accuracy_index': 2,  'accuracy': 0.93,  'head_removed': 475},
 {'noise_index': 6,  'accuracy_index': 2,  'accuracy': 0.94,  'head_removed': 475},
 {'noise_index': 7,  'accuracy_index': 4,  'accuracy': 0.88,  'head_removed': 617},
 {'noise_index': 8,  'accuracy_index': 4,  'accuracy': 0.95,  'head_removed': 617},
 {'noise_index': 9,  'accuracy_index': 5,  'accuracy': 0.91,  'head_removed': 663}]
import pickle
with open(f'/home//StepByStep/results/reasoning/combined/normalised_combined_matrix.pkl', 'rb') as f:
    patched_head = pickle.load(f)
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
for i in range(8):
    left_idx = max_range_idx - i
    right_idx = max_range_idx + i
    if(left_idx < 0):
        left_idx = 0
    if(right_idx >= len(hist)):
        right_idx = len(hist) - 1
    threshold_ranges.append([bin_edges[left_idx], bin_edges[right_idx]])
# %%

# %%

# %%
import argparse
parser = argparse.ArgumentParser()
# python3 IndividualHeadAccuracy.py --noiseIndex 0 --device "cuda:1" --numExamples 10 --alteranteExamples 10
parser.add_argument("-noiseIndex", "--noiseIndex", help = "noiseIndex")
parser.add_argument("-start", "--start", help = "start")
parser.add_argument("-end", "--end", help = "end")
parser.add_argument("-device", "--device", help = "device")
parser.add_argument("-example_number", "--example_number", help = "example_number")


args = parser.parse_args()
# %%
noise_index = int(args.noiseIndex)
example_number = int(args.example_number)
start = int(args.start)
end = int(args.end)
# noise_index = 3
# example_number = 3
# start = 0
# end = 6
print(f"Noise index {noise_index}")
print(f"Example number {example_number}")
print(f"Start {start}")
print(f"End {end}")
import pickle
with open(f'/home//StepByStep/results/reasoning/combined/{noise_index}/accuracy/Activation_example_start_{start}_end_{end}.pickle', 'rb') as f:
    data = CPU_Unpickler(f).load()

# %%
threshold_range = threshold_ranges[reasoning_heads[noise_index]['accuracy_index']]
def getImportantHeads(threadShold, matrix):
    importantHeads = []
    for layer in range(len(matrix)):
        for head in range(len(matrix[layer])):
            if(matrix[layer][head] > threadShold[1] or matrix[layer][head] < threadShold[0]):
                importantHeads.append((layer, head))
    return sorted(importantHeads)
importantHeads = getImportantHeads(threshold_range, patched_head)

print(f'Important heads {importantHeads}')
print(f'Lenght of important heads {len(importantHeads)}')
# %%
data.keys()

# %%

patched_logits = data['patched_logits']
patched_cache = data['patched_cache']
input_id = data['input_ids'][example_number]

# %%
device = args.device
# device = 'cuda:1'


# %%
import sys
sys.path.append('../')

# %%
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from utilsFile.loadModel import loadTransformerLensModel, runCacheActivationPatching
MODEL_PATH = '/home/models//Llama-2-7b-hf'
# modelPath='/home/models/vicuna-7b'
model, tokenizer = loadTransformerLensModel(MODEL_PATH)
model = model.to(device)

# %%
l = 4
k = 1
p = 1

# %%
umembed = model.unembed
ln_final = model.ln_final

# %%
def stage_I(pos, token_id, layer_number):
    from tqdm import tqdm
    prob_each_layer_head = []
    avg_prob = 0
    for layer in tqdm(range(layer_number), desc = f'stage_I layer {layer_number}'):
        for head in range(32):
            z: Float[Tensor, "batch seq d_head"] = patched_cache[utils.get_act_name("z", layer, "attn")][:, :, head].to(device)
            N = z.size(0)
            output: Float[Tensor, "batch seq d_model"] = z @ model.W_O[layer, head]
            output = umembed(ln_final(output))
            output_end_token_prob = torch.softmax(output[:, pos, :], dim=-1)[example_number,token_id]
            prob_each_layer_head.append({'layer': layer, 'head': head, 'prob': output_end_token_prob.item()})
            avg_prob += output_end_token_prob.item()
    # avg_prob /= (32*layer_number)
    
    top_l = sorted(prob_each_layer_head, key=lambda x: x['prob'], reverse=True)
    top_l_layer_head = []
    for x in top_l:
        if((x['layer'], x['head']) in importantHeads):
            top_l_layer_head.append({'layer' : x['layer'], 'head' : x['head'], 'input_id' : token_id, 'pos' : pos, 'prob' : x['prob']})
    top_l_layer_head = top_l_layer_head[:l]
    print("top l layer heads")
    print(top_l_layer_head)
    return top_l_layer_head


# %%
def stage_II(layer, head, position):
    attn_probs: Float[Tensor, "batch q k"] = patched_cache[utils.get_act_name("pattern", layer)][:, head].detach().cpu().numpy()[example_number][position,:]
    median_attn_probs = np.median(attn_probs)
    # print(attn_probs)
    attn_prob_with_input_id = [{'prob': attn_probs[i], 'input_id': input_id[i], 'pos': i} for i in range(len(attn_probs))]
    top_k_ids = []
    for d in attn_prob_with_input_id:
        if(d['prob'] > median_attn_probs):
            top_k_ids.append(d)
    top_k_ids = sorted(top_k_ids, key=lambda k: k['prob'], reverse=True)[:k]
    print("top K ids")
    print(top_k_ids)
    return top_k_ids



# %%
def stage_III(layer, pos):
    """
        Layer : layer number
        pos : position of the residual stream to look at
    Returns the top p tokens for residual stream at layer and position
    """
    residual_pre = patched_cache[utils.get_act_name("resid_pre", layer)].to(device)

    residual_pre = umembed(ln_final(residual_pre))
    residual_pre_pos_token_prob = torch.softmax(residual_pre[:, pos, :], dim=-1)[example_number]
    median_residual_pre_pos_token_prob = torch.median(residual_pre_pos_token_prob)
    sorted_residual_pre_pos_token_prob = torch.sort(residual_pre_pos_token_prob, descending=True)
    top_p_ids_data = []
    for i in range(len(sorted_residual_pre_pos_token_prob.indices)):
        if(sorted_residual_pre_pos_token_prob.values[i].item() > median_residual_pre_pos_token_prob.item()):
            top_p_ids_data.append({'prob': sorted_residual_pre_pos_token_prob.values[i].item(), 'input_id': sorted_residual_pre_pos_token_prob.indices[i].item()})
            if(len(top_p_ids_data) == p):
                break
    print("top p ids")
    print(top_p_ids_data)
    return top_p_ids_data

# %%
from tqdm import tqdm
def combined_steps(layer_number, head_number, position):
    top_k_ids = stage_II(layer_number, head_number, position)
    print(f'Top K positions founded for {layer_number} and {head_number}, length {len(top_k_ids)}')
    filtered_layer_head = []
    for d in tqdm(top_k_ids):
        top_p_ids = stage_III(layer_number, d['pos'])
        print(f'Top p tokens founded for {layer_number} and {d["pos"]}, length {len(top_p_ids)}')
        for d2 in top_p_ids:
            print(f"Running stage I for {layer_number} and position {d['pos']}, token {d2['input_id']}")
            layer_head_data_I = stage_I(d['pos'], d2['input_id'], layer_number)
            for d3 in layer_head_data_I:
                if(d3['layer'] < layer_number):
                    filtered_layer_head.append(d3)
    return filtered_layer_head


# %%
token_X_id = torch.argmax(patched_logits[:, -1, :], dim=1)[example_number]
token_X = tokenizer.decode(token_X_id)

initial_layer_head = stage_I(-1, token_X_id, 32)
run = True
depth = 0
information_flow_result = []

# %%
layer_head_pos_done = []
while(run):
    all_layer_0 = True
    attending_layer_head_full = []
    for layer_head in tqdm(initial_layer_head, desc=f'initial_layer_head at depth {depth}'):
        layer = layer_head['layer']
        head = layer_head['head']
        pos = layer_head['pos']
        if((layer, head, pos) in layer_head_pos_done):
            print("skipping")
            continue
        print(f'H head at {layer} and {head}, depth {depth}')
        if(layer != 0):
            all_layer_0 = False
        attending_layer_head = combined_steps(layer, head, pos)
        information_flow_result.append({'layer': layer, 'head' : head, 'pos' : pos, 'depth' : depth, 'attending_layer_head': attending_layer_head})
        attending_layer_head_full.extend(attending_layer_head)
        layer_head_pos_done.append((layer, head, pos))
    depth += 1
    if(l > 1):
        l -= 1
    if(k > 1):
        k -= 1
    if(p > 1):
        p -= 1
    attending_layer_head_full = [(x['layer'], x['head'], x['pos']) for x in attending_layer_head_full]
    attending_layer_head_full = [{'layer' : x[0], 'head' : x[1], 'pos' : x[2]} for x in list(set(attending_layer_head_full))]
    initial_layer_head = attending_layer_head_full
    if(all_layer_0):
        print("All at layer 0 or no previous head detected")
        run = False
    
    import os
    save_dir = f'result/{noise_index}'
    os.makedirs(save_dir, exist_ok=True)
    import pickle
    with open(f'{save_dir}/example_{start + example_number}.pickle', 'wb') as f:
        pickle.dump(information_flow_result, f)

# %%
# from tqdm import tqdm
# information_flow_pairs = []
# layer_head = stage_I(-1, token_X_id, 32,32)
# for layer, head in layer_head:
#     top_k_ids = stage_II(layer, head)
#     filtered_layer_head_I = []
#     for d in tqdm(top_k_ids):
#         top_p_ids = stage_III(layer, d['pos'])
#         for d2 in top_p_ids:
#             layer_head_data_I = stage_I(d['pos'], d2['input_id'], layer, 32)
#             for d3 in layer_head_data_I:
#                 if(d3['layer'] < layer):
#                     filtered_layer_head_I.append(d3)
#     information_flow_pairs.append({'init_layer_head' : (layer, head), 'final_layer_head': filtered_layer_head_I})



# # %%
# top_k_ids

# # %%
# from tqdm import tqdm
# umembed = model.unembed
# ln_final = model.ln_final
# avg_prob = 0
# prob_each_layer_head = []
# for layer in tqdm(range(32)):
#     for head in range(32):
#         z: Float[Tensor, "batch seq d_head"] = patched_cache[utils.get_act_name("z", layer, "attn")][:, :, head].to(device)
#         N = z.size(0)
#         output: Float[Tensor, "batch seq d_model"] = z @ model.W_O[layer, head]
#         output = umembed(ln_final(output))
#         output_end_token_prob = torch.softmax(output[:, -1, :], dim=-1)[example_number, token_X_id]
#         prob_each_layer_head.append({'layer': layer, 'head': head, 'prob': output_end_token_prob.item()})
# avg_prob /= 1024
        

# # %%
# # sort the prob_each_layer_head dict based on prob
# prob_each_layer_head = sorted(prob_each_layer_head, key=lambda k: k['prob'], reverse=True)

# # %%
# h_x = [{'layer': x['layer'], 'head': x['head']} for x in prob_each_layer_head[:l]]

# # %%


# # %%


# # %%
# for dict in h_x:
#     layer  = dict['layer']
#     head = dict['head']
#     attn_probs: Float[Tensor, "batch q k"] = patched_cache[utils.get_act_name("pattern", layer)][:, head].detach().cpu().numpy()[example_number][-1,:]
#     mean_attn_probs = np.mean(attn_probs)
#     attn_prob_with_input_id = [{'prob': attn_probs[i], 'input_id': input_id[i]} for i in range(len(attn_probs))]
#     # top_k_ids = []
#     # for d in prob_with_input_id:
#     #     if(d['prob'] > mean_attn_probs):
#     #         top_k_ids.append(d)
#     top_k_ids = sorted(attn_prob_with_input_id, key=lambda k: k['prob'], reverse=True)[:k]
#     residual_pre = patched_cache[utils.get_act_name("resid_pre", layer)]
#     residual_pre = umembed(ln_final(residual_pre))
#     residual_pre_end_token_prob = torch.softmax(output[:, -1, :], dim=-1)[example_number]

#     residul_prob_with_input_id = [{'prob': residual_pre_end_token_prob[i], 'input_id': input_id[i]} for i in range(len(residual_pre_end_token_prob))]
#     top_p_ids = sorted(residul_prob_with_input_id, key=lambda k: k['prob'], reverse=True)[:p]
    




# # %%
# attn_probs.shape

# # %%


# # %%


# # %%
# def imshow(tensor, renderer=None, save='head.html', **kwargs):
#     fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs)
#     fig.show(renderer)
#     # fig.write_html(save)

# # %%
# imshow(prob_each_layer_head, save=f'first_projection.html', renderer='browser', labels={"x":"Head", "y":"Layer"})

# # %%


# # %%


# # %%
# import numpy as np

# def kendall_w(expt_ratings):
#     if expt_ratings.ndim!=2:
#         raise 'ratings matrix must be 2-dimensional'
#     m = expt_ratings.shape[0] #raters
#     n = expt_ratings.shape[1] # items rated
#     denom = m**2*(n**3-n)
#     rating_sums = np.sum(expt_ratings, axis=0)
#     S = n*np.var(rating_sums)
#     return 12*S/denom

# informative = np.array([[3,4,3,2,4,4,5,4,2,4,5,4,5,4,3],[4,3,3,2,3,3,4,4,4,5,4,5,4,5,5],[4,2,2,2,2,4,3,4,2,3,4,4,4,4,4],[4,4,3,2,4,5,5,4,2,4,5,4,5,3,3]])
# concise = np.array([[4,3,3,2,4,4,5,4,3,3,3,3,4,4,3],[2,4,4,3,3,3,3,3,5,3,3,3,3,4,3],[2,3,4,4,3,4,4,3,4,4,3,4,4,4,3],[3,5,4,3,4,3,5,4,4,3,1,2,3,4,2]])
# fluent = np.array([[4, 4, 3, 4, 4, 5, 5, 3, 4, 4, 4, 5, 5, 4, 4],[2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4],[5, 3, 5, 4, 4, 4, 4, 4, 5, 4, 3, 4, 3, 4, 4],[4, 4, 2, 4, 3, 4, 4, 2, 4, 4, 4, 5, 5, 4, 4]])
# entity = np.array([[3,2,3,2,3,4,4,3,2,4,5,4,5,3,4],[4,3,4,3,4,2,4,5,4,4,4,5,4,4,5],[4,1,3,1,1,4,3,4,2,3,4,4,4,4,4],[5,5,3,1,3,4,5,3,1,3,5,3,5,3,4]])

# ratings = []
# for i in range(len(informative)):
#     ratings.append(entity[i])
# ratings = np.array(ratings)
# W = kendall_w(ratings)

# # %%
# avg_res = 0
# count = 0
# for i in range(4):
#     for j in range(i+1, 4):
#         x = ratings[i]
#         y = ratings[j]
#         print(i, j)
#         from scipy import stats
#         res = stats.spearmanr(x, y)
#         avg_res += res.statistic
#         count += 1

# # %%
# avg_res /count

# # %%
# informative : 0.41329052450374165
# concise : 0.16414525002413802
# fluent : 0.01443070029536613
# entity : 0.3611964031741075


# # %%
# W

# # %%
# information : 0.028035714285714282
# concise : 0.01330357142857143
# fluent : 0.006011904761904762
# entity : 0.03279761904761905

# # %%


# # %%


# # %%



