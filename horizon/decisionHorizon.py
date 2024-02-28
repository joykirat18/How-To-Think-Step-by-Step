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
with open('../data/activationPatching_llama2_horizon.json', 'r') as f:
    COTData = json.load(f)

# %%
# import argparse
# parser = argparse.ArgumentParser()
# # python3 IndividualHeadAccuracy.py --noiseIndex 0 --device "cuda:1" --numExamples 10 --alteranteExamples 10
# parser.add_argument("-noiseIndex", "--noiseIndex", help = "noiseIndex")
# parser.add_argument("-device", "--device", help = "device")
# parser.add_argument("-numExamples", "--numExamples", help = "numExamples")
# parser.add_argument("-alternateExamples", "--alternateExamples", help = "alternateExamples")

# # alternateExamples
# parser.add_argument("--generate", action="store_true", help = "generate")
# # parser.add_argument("-modelPath", "--modelPath", help = "modelPath")
# parser.add_argument("--combined", action="store_true", help = "combined")
# # parser.add_argument("-combine_type", "--combine_type", help = "combine_type")

# args = parser.parse_args()
# %%
import sys
sys.path.append('../')
from utilsFile.fewShotConstants import TemplateCOT_fictional, TemplateCOT_false

# %%
device = "cuda:2"
# device = args.device if torch.cuda.is_available() else "cpu"

# %%

from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from utilsFile.loadModel import loadTransformerModel, runCacheActivationPatching
MODEL_PATH = '/home/models/Llama-2-7b-hf'
# modelPath='/home/models/vicuna-7b'
model, tokenizer = loadTransformerModel(MODEL_PATH)
model = model.to(device)
# %%

# %%
prompts = [data['prompt'] for data in COTData]
noisePrompts = [data['horizonNoisePrompt'] for data in COTData]
answers = [data['answer'] for data in COTData]

# %%
stopType = '.' # ▁is .
# %%
result = []
from tqdm import tqdm
for i in tqdm(range(len(prompts))):
    prompt = TemplateCOT_fictional.format(prompts[i])
    noise_prompt = TemplateCOT_fictional.format(noisePrompts[i])
    input_id = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    noise_input_id = tokenizer(noise_prompt, return_tensors="pt").input_ids.to(device)
    max_new_tokens = 100
    original_input_id = input_id.clone()
    original_noise_input_id = noise_input_id.clone()
    stopped = False
    while(max_new_tokens > 0):
        logit = model(input_id)[0]
        next_token_id = torch.argmax(logit[:, -1, :], dim=-1)
        next_token = tokenizer.convert_ids_to_tokens(next_token_id)[0]
        input_id = torch.cat([input_id, next_token_id.unsqueeze(1)], dim=-1)
        # print(tokenizer.decode(input_id[0][300:], skip_special_tokens=True))
        noise_input_id = torch.cat([noise_input_id, next_token_id.unsqueeze(1)], dim=-1)
        print(next_token)
        if(next_token == '<0x0A>'):
            break
        if(next_token == stopType and stopped == False):
            stopped = True
            input_id = noise_input_id
            print('stopped')
        max_new_tokens -= 1
    generated_text = tokenizer.decode(input_id[0][original_noise_input_id.size(1):], skip_special_tokens=True)
    print(f'generated : {generated_text}')
    print(f'answer : {answers[i]}')

    result.append({'prompt' : prompts[i], 'noise_prompt' : noisePrompts[i], 'generated_text' : generated_text, 'answer' : answers[i]})
    with open(f'horizonLLAMA2({stopType}).json', 'w') as f:
        json.dump(result, f)
    del input_id, noise_input_id, original_input_id, original_noise_input_id

# %%
