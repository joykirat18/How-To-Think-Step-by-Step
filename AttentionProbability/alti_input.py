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
with open(f'../results/reasoning/combined/normalised_combined_matrix.pkl', 'rb') as f:
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
import pickle
import argparse
parser = argparse.ArgumentParser()
# python3 IndividualHeadAccuracy.py --noiseIndex 0 --device "cuda:1" --numExamples 10 --alteranteExamples 10
# parser.add_argument("-firstHead", "--firstHead")
# parser.add_argument("-secondHead", "--secondHead")
parser.add_argument("-noiseIndex", "--noiseIndex")
# parser.add_argument("-type", "--type")
args = parser.parse_args()

noise_index = int(args.noiseIndex)
with open(f'/home/joykirat/JS/StepByStep/results/reasoning/combined/{noise_index}/accuracy_noise_input_context/ALTI_score.pickle', 'rb') as f:
    ALTI_score = pickle.load(f)

# %%
altiScore = ALTI_score['prob_distribution']
input_ids = ALTI_score['input_ids']

# %%
threshold_range = threshold_ranges[reasoning_heads[noise_index]['accuracy_index']]

# %%
def getImportantHeads(threadShold, matrix):
    importantHeads = []
    for layer in range(len(matrix)):
        for head in range(len(matrix[layer])):
            if(matrix[layer][head] > threadShold[1] or matrix[layer][head] < threadShold[0]):
                importantHeads.append((layer, head))
    return sorted(importantHeads)

# %%
importantHeads = getImportantHeads(threshold_range, patched_head)

# %%
importantHeads[0]

# %%
layer = 0
head = 0
number = 0
layer_head_distribution = altiScore[layer][head][number]
input_id = input_ids[number]


# %%
from tqdm import tqdm
layer_head_data = []
min_length = 100000
for pair in tqdm(importantHeads):
    layer = pair[0]
    head = pair[1]
    non_padded_distributions = []
    non_padded_input_ids = []
    for number in range(len(input_ids)):
        layer_head_distribution = altiScore[layer][head][number]
        input_id = input_ids[number]
        non_padded_distribution = []
        non_paded_input_id = []
        for i in range(len(input_id)):
            if(input_id[i] != 1):
                non_padded_distribution.append(layer_head_distribution[i])
                non_paded_input_id.append(input_id[i].item())
        non_padded_distributions.append(non_padded_distribution)
        non_padded_input_ids.append(non_paded_input_id)
        if(len(non_padded_distribution) < min_length):
            min_length = len(non_padded_distribution)
    layer_head_data.append({'layer': layer, 'head': head, 'distribution': non_padded_distributions, 'input_ids': non_padded_input_ids})
        

# %%
min_length

# %%
MODEL_PATH = '/home/models/joykirat/Llama-2-7b-hf'
from transformers import LlamaForCausalLM, LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)

# %%
input_id_0 = layer_head_data[0]['input_ids'][0]
input_id_1 = layer_head_data[0]['input_ids'][1]
common_length = 0
for i in range(min_length):
    if(input_id_0[i] == input_id_1[i]):
        common_length = i
    else:
        break

# %%
import pandas as pd
import plotly.express as px
def line(tensor, label, renderer=None, xaxis="", yaxis="", hover_labels=None, **kwargs):
    # Create a DataFrame with x and y data
    df = pd.DataFrame({'x': label, 'y': tensor})
    
    # Create a figure using plotly express
    fig = px.line(df, x='x', y='y', labels={"x": xaxis, "y": yaxis}, hover_data=hover_labels, **kwargs)
    
    # Show the plot
    fig.show(renderer)
    # import os
    # fig.write_html(f'{save_dir}/residualProb.html')

# %%
save_dir = f'IndividualAlti_noise_input_context/{noise_index}'
import os
os.makedirs(save_dir,exist_ok=True)


# %%
import pandas as pd
import plotly.express as px
def scatter(tensor, label,number, renderer=None, xaxis="", yaxis="", hover_labels=None, **kwargs):
    # Create a DataFrame with x and y data
    df = pd.DataFrame({'x': label, 'y': tensor})
    
    # Create a figure using plotly express
    fig = px.scatter(df, x='x', y='y', labels={"x": xaxis, "y": yaxis}, **kwargs)
    
    # Show the plot
    fig.update_layout(width=1600)
    # fig.show(renderer)

    fig.write_html(f'{save_dir}/example_{number}.html')

# %%
len(input_ids[number]) - common_length

# %%
len(input_ids[2]) - common_length

# %%
from tqdm import tqdm
layer_head_data = []
min_length = 100000

# print(len())
for number in range(11):
    input_id = input_ids[number]
    combined_distribution_length = 0
    for i in range(len(input_id)):
        if(input_id[i] != 1):
            combined_distribution_length += 1
    combined_distribution = np.array([0.0] * (combined_distribution_length - common_length))
    for pair in tqdm(importantHeads):
        layer = pair[0]
        head = pair[1]
        
        input_id = input_ids[number]
        non_padded_distribution = []
        non_paded_input_id = []
        label = []
        layer_head_distribution = altiScore[layer][head][number]
        for i in range(len(input_id)):
            if(input_id[i] != 1):
                non_padded_distribution.append(layer_head_distribution[i])
                non_paded_input_id.append(input_id[i].item())
                label.append(tokenizer.decode([input_id[i].item()]) + f'_{i}')
        # print(len(non_padded_distribution[common_length:]))
        combined_distribution += np.array(non_padded_distribution[common_length:])
    combined_distribution /= len(importantHeads)
    scatter(np.array(combined_distribution), label[common_length:], number, xaxis='Token', yaxis='Probability', title=f'Noise Index : {noise_index}, example : {number}')

# %%



