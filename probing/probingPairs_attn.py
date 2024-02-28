# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast
import torch
import sys
import os
from peft import PeftModel, PeftConfig
from peft import LoraConfig, TaskType, get_peft_model

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from transformers import LlamaForCausalLM, LlamaTokenizer


sys.path.append('../../JS')
from honest_llama import llama
# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-numberOfShots", "--numberOfShots", help = "numberOfShots")
parser.add_argument("-device", "--device", help = "device")
parser.add_argument("-type", "--type", help = "type")

args = parser.parse_args()
# %%
path_to_model = '/home/models//Llama-2-7b-hf'  # push to runtime argument
path_to_tokenizer = '/home/models//Llama-2-7b-hf' # push to runtime argument
# ../frugal_lms/saved_model/base_gptj/
# %%
# device = 'cuda:0' # push it to runtime argument
device = args.device
# mt = ModelAndTokenizer(path_to_model, device=device)
# tokenizer = LlamaTokenizerFast.from_pretrained('hf-internal-testing/llama-tokenizer', is_fast=True)
# model = AutoModelForCausalLM.from_pretrained(path_to_model, low_cpu_mem_usage=True).to(device)

tokenizer = LlamaTokenizer.from_pretrained(path_to_tokenizer)
model = llama.LLaMAForCausalLM.from_pretrained(path_to_model, low_cpu_mem_usage=True).to(device)

# %%


numberOfShots = int(args.numberOfShots)
typeTrain = args.type
# numberOfShots = 1
directory = f"llama2_{typeTrain}/{numberOfShots}shot"
os.makedirs(directory, exist_ok=True)
os.makedirs(f'{directory}/chunks', exist_ok=True)

import json
# with open('data/2hop_250_QA_pair_distractor.json', 'r') as f:
#     data = json.load(f)

shots = ""

with open('data/fewShotPrompts.json', 'r') as f:
    fewShots = json.load(f)


for i in range(numberOfShots):
    question = fewShots[i]['question'] + ' ' + fewShots[i]['query']
    shots += f"### Input: \n{question} Let us think step by step.\n### Response: \n{fewShots[i]['answer']}\n\n"

# %%
import json
with open(f'data/2Hop_{typeTrain}_data.json', 'r') as f:
    data = json.load(f)
# data = data[:1000]
# %%
from transformers import pipeline, set_seed, StoppingCriteriaList, StoppingCriteria
import torch
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, tokenizer=None,stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.tokenizer=tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # print(input_ids)
        full_sentence = self.tokenizer.decode(input_ids[0].tolist(), clean_up_tokenization_spaces=True).strip()
        last_token = self.tokenizer.decode(input_ids[0][-1].tolist(), clean_up_tokenization_spaces=True)
        # print(last_token)
        if(last_token == '<0x0A>'):
            return True
        return False
    
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer=tokenizer,stops=['stop_words'])])

# %%
class TreeNode:
    def __init__(self, name, children=None):
        self.name = name
        self.parentName = None
        self.children = children if children else []

def parseTreeString(tree_structure):

    current_node = TreeNode('root')
    topNode = current_node
    # print(topNode.name)
    def extractPropertyAndNode(line):
        property_name = None
        if('properties:' in line):
            property_name = line.split(':')[-1].strip().replace(' ', '')
            node_name = line.split('properties:')[0].strip().replace('(', '').replace(')', '')
        else:
            node_name = line.strip().replace('(', '').replace(')', '')
        return node_name, property_name
    # Split the tree structure into lines
    lines = tree_structure.split("\n")
    # print(len(lines))
    for line in lines:
        line = line.strip("()")

        node_name, property_name = extractPropertyAndNode(line)
        # print(node_name, property_name)
        if(node_name == ''):
            continue
        if(property_name is None):
            property_name = ''

        name_node = TreeNode(node_name)
        property_node = TreeNode(property_name)

        if(current_node is None):
            current_node = name_node
            if(property_name != ''):
                current_node.children.append(property_node)
        else:
            current_node.children.append(name_node)
            current_node = name_node
            if(property_name != ''):
                current_node.children.append(property_node)
                
    topNode = topNode.children[0]

    return topNode
def create_adjacency_matrix(root):
    nodes = get_nodes(root)
    num_nodes = len(nodes)
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    node_indices = {node: index for index, node in enumerate(nodes)}
    fill_adjacency_matrix(root, adjacency_matrix, node_indices)
    return adjacency_matrix


def get_nodes(root):
    nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children)
    return nodes


def fill_adjacency_matrix(node, adjacency_matrix, node_indices):
    node_index = node_indices[node]
    for child in node.children:
        child_index = node_indices[child]
        adjacency_matrix[node_index][child_index] = 1
        fill_adjacency_matrix(child, adjacency_matrix, node_indices)

def create_adjacency_matrix(root):
    nodes = get_nodes(root)
    num_nodes = len(nodes)
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    node_indices = {node: index for index, node in enumerate(nodes)}
    fill_adjacency_matrix(root, adjacency_matrix, node_indices)
    return adjacency_matrix


def get_nodes(root):
    nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children)
    return nodes


def fill_adjacency_matrix(node, adjacency_matrix, node_indices):
    node_index = node_indices[node]
    for child in node.children:
        child_index = node_indices[child]
        adjacency_matrix[node_index][child_index] = 1
        fill_adjacency_matrix(child, adjacency_matrix, node_indices)

# %%
import pickle
with open('data/pluralToNouns.pickle', 'rb') as handle:
    pluralToNouns = pickle.load(handle)
def preprocessing(word):
    word = word.lower()
    # print(word)
    if(word in pluralToNouns):
        word = pluralToNouns[word]
    return word
    

# %%
def tokenize_sentence_with_parent(sentence, tokenizer):
    tokenized_sentence = tokenizer.tokenize(sentence)
    # print(tokenized_sentence)
    # print(tokenized_sentence)
    parent_words = []

    i = 0
    current_word = ""

    for i in range(len(tokenized_sentence)):
        if(tokenized_sentence[i] == '.'):
            parent_words.append('.')
        # print(tokenized_sentence[i].startswith("▁"))
        elif tokenized_sentence[i].startswith("▁") and (i + 1 < len(tokenized_sentence)) and tokenized_sentence[i + 1].startswith("▁"):
            parent_words.append(tokenized_sentence[i].replace("▁", ""))
        elif(tokenized_sentence[i].startswith("▁") and (i  == (len(tokenized_sentence) - 1))):
            parent_words.append(tokenized_sentence[i].replace("▁", ""))
        elif(tokenized_sentence[i].startswith("▁") == True and (i + 1 < len(tokenized_sentence)) and tokenized_sentence[i + 1].startswith("▁") == False):
            end = i +  1
            while(end < len(tokenized_sentence)):
                if(tokenized_sentence[end].startswith("▁") == True or tokenized_sentence[end] == '.'):
                    end -=  1
                    break
                end += 1
            start = i
            notPresent = False
            while(start > 0):
                if(tokenized_sentence[start] == '.'):
                    start = i
                    break
                if(tokenized_sentence[start] == '▁not'):
                    notPresent = True
                    break
                start -= 1
            if(notPresent == False):
                start = i

            parent_word = ""
            # print(end)
            for j in range(start, end + 1):
                if(j < len(tokenized_sentence) - 1):
                    parent_word += tokenizer.convert_tokens_to_string([tokenized_sentence[j]])
            # print("parent word: ", parent_word)
            # print(start, end)
            parent_words.append(parent_word)

             
        elif(tokenized_sentence[i].startswith("▁") == False):
            temp = []
            start = i
            # print(tokenized_sentence[i])
            while(start > 0):
                if(tokenized_sentence[start].startswith("▁") == True or tokenized_sentence[start] == '.'):
                    break
                start -= 1
            
            start2 = i
            notPresent = False
            while(start2 > 0):
                if(tokenized_sentence[start2] == '.'):
                    start2 = i
                    break
                if(tokenized_sentence[start2] == '▁not'):
                    notPresent = True
                    break
                start2 -= 1
            if(notPresent == True):
                start = start2
            # print("start: ",tokenized_sentence[start])
            end = i
            while(end < len(tokenized_sentence)):
                if(tokenized_sentence[end].startswith("▁") == True or tokenized_sentence[end] == '.'):
                    end -=  1
                    break
                end += 1
            # print("end: ",tokenized_sentence[end])
            parent_word = ""
            for j in range(start, end + 1):
                if(j < len(tokenized_sentence) - 1):
                # print(tokenized_sentence[j])
                    parent_word += tokenizer.convert_tokens_to_string([tokenized_sentence[j]])
            # print("parent word: ", parent_word)
            parent_words.append(parent_word)
    for i in range(len(parent_words)):
        parent_words[i] = preprocessing(parent_words[i])
    # print(tokenized_sentence)
    # print(parent_words)
    return tokenized_sentence, parent_words

def findIndexInMatrix(nodes, word, parent_word):
    for i in range(len(nodes)):
        if(nodes[i].name == parent_word):
            return i
    return None

def findIndexInMatrix(nodes, word, parent_word):
    for i in range(len(nodes)):
        if(nodes[i].name == parent_word):
            return i
    return None


# %%
def find_indices(larger_array, smaller_array):
    larger_len = len(larger_array)
    smaller_len = len(smaller_array)
    # print(larger_array)
    # print(smaller_array)
    for i in range(larger_len - smaller_len + 1):
        if larger_array[i:i+smaller_len] == smaller_array:
            return i - 2, i+smaller_len-1

    return None, None

# %%
import numpy as np
def adjacency_to_distance(adj_matrix):
    num_vertices = len(adj_matrix)
    distance_matrix = np.full((num_vertices, num_vertices), float('inf'))

    for i in range(num_vertices):
        distance_matrix[i][i] = 0

    for i in range(num_vertices):
        for j in range(num_vertices):
            if adj_matrix[i][j] != 0:
                distance_matrix[i][j] = adj_matrix[i][j]

    for k in range(num_vertices):
        distance_matrix = np.minimum(distance_matrix, distance_matrix[:, k:k+1] + distance_matrix[k:k+1, :])

    return distance_matrix.tolist()

def getUnRelatedPair(tokenized_sentence, parent_words, adjacency_matrix, allNodeEntities, typeA, typeB):
    unrelatedPair = []
    start = 0
    end = len(tokenized_sentence)
    if(typeA == 'first'):
        start = 1
    if(typeA == 'last'):
        end = len(tokenized_sentence) - 1
    for i in range(start, end):
        if(typeA == 'first'):
            if(parent_words[i - 1] == parent_words[i]):
                continue
        elif(typeA == 'last'):
            if(parent_words[i] == parent_words[i + 1]):
                continue
        for j in range(i, len(tokenized_sentence)):
            if(typeB == 'last'):
                if(parent_words[i] != parent_words[j] and parent_words[i] in allNodeEntities and parent_words[j] in allNodeEntities and adjacency_matrix[i][j] == float('inf')
                   and j < (len(tokenized_sentence) - 1) and parent_words[j] != parent_words[j + 1]):
                    # print(tokenized_sentence[i], tokenized_sentence[j], parent_words[i], parent_words[j])
                    unrelatedPair.append((i, j))
                    break

            else:
                if(parent_words[i] != parent_words[j] and parent_words[i] in allNodeEntities and parent_words[j] in allNodeEntities and adjacency_matrix[i][j] == float('inf')):
                    # print(tokenized_sentence[i], tokenized_sentence[j], parent_words[i], parent_words[j])
                    unrelatedPair.append((i, j))
                    if(typeB == 'first'):
                        break
    return unrelatedPair

def getPositiveAndNegativePair(tokenized_sentence, parent_words, matrix, typeA, typeB):
    positivePair = []
    negativePair = []
    start = 0
    end = len(tokenized_sentence)
    for i in range(start, end):
        if(typeA == 'first'):
            if(i >= 0 and parent_words[i - 1] == parent_words[i]):
                continue
        elif(typeA == 'last'):
            if(i < (len(tokenized_sentence) - 1) and parent_words[i] == parent_words[i + 1]):
                continue
        j = i
        while(j < len(tokenized_sentence)):
            if(tokenized_sentence[j] == '.'):
                break
            if(typeB == 'last'):
                if(matrix[i][j] == 1 and j < (len(tokenized_sentence) - 1) and matrix[i][j + 1] != 1):
                    if('not' in parent_words[j]):
                        # print(tokenized_sentence[i], tokenized_sentence[j], parent_words[i], parent_words[j])
                        negativePair.append((i, j))
                    else:
                        # print(tokenized_sentence[i], tokenized_sentence[j], parent_words[i], parent_words[j])
                        positivePair.append((i, j))
                    break
            else:
                if(matrix[i][j] == 1):
                    if('not' in parent_words[j]):
                        # print(tokenized_sentence[i], tokenized_sentence[j], parent_words[i], parent_words[j])
                        negativePair.append((i, j))
                    else:
                        # print(tokenized_sentence[i], tokenized_sentence[j], parent_words[i], parent_words[j])
                        positivePair.append((i, j))
                    if(typeB == 'first'):
                        break
            j += 1
    return positivePair, negativePair

def getPairs(tokenized_sentence, parent_words, matrix, allNodeEntities, typeA, typeB):
    positivePair, negativePair = getPositiveAndNegativePair(tokenized_sentence, parent_words, matrix, typeA, typeB)
    distanceMatrix = adjacency_to_distance(matrix)
    unrelatedPair = getUnRelatedPair(tokenized_sentence, parent_words, distanceMatrix, allNodeEntities, typeA, typeB)
    consecutivePair = []
    for i in range(len(tokenized_sentence) - 1):
        consecutivePair.append((i, i + 1))
    randomPair = []
    for i in range(len(tokenized_sentence)):
        for j in range(len(tokenized_sentence)):
            if(i != j):
                randomPair.append((i, j))
    import random
    random.shuffle(randomPair)
    randomPair = randomPair[:len(randomPair) // 2]

    

    return positivePair, negativePair, unrelatedPair, consecutivePair, randomPair

types = ['first']

# %%
import numpy as np
from sklearn.decomposition import PCA
def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
from tqdm import tqdm
# %%
allSimiliarity = []
chunk_number = 0
for entry in tqdm(range(len(data))):
    try:
        # entry = 0
        question = data[entry]['question'] + ' ' + data[entry]['query']
        tree_structure = data[entry]['tree']
        nShots = shots + f"### Input : \n{question} Let us think step by step.\n### Response :\n"
        # Twoshot = f"### Input : \nEvery phorpus is a twimpus. Phorpuses are kergits. Every phorpus is wooden. Each jempor is a phorpus. Jempors are borpins. Each jempor is happy. Every kergit is brown. Each borpin is not bright. Jempuses are not wooden. Every jelgit is a storpist. Jelgits are opaque. Rex is a jelgit. Rex is a jempor. True or false: Rex is wooden. Let us think step by step.\n### Response : \nRex is a jempor. Each jempor is a phorpus. Rex is a phorpus. Every phorpus is wooden. Rex is wooden. True\n\n### Input : \nBorpins are daumpins. Each borpin is a jelgit. Every borpin is not wooden. Phorpuses are wooden. Jempors are borpins. Every jempor is a jempus. Jempors are purple. Jelgits are transparent. Every jempus is happy. Every storpist is a twimpus. Storpists are not small. Sam is a storpist. Sam is a jempor. True or false: Sam is wooden. Let us think step by step.\n### Response : \nSam is a jempor. Jempors are borpins. Sam is a borpin. Every borpin is not wooden. Sam is not wooden. False\n\n### Input : \n{question} Let us think step by step.\n### Response :\n"
        # Twoshot = f"### Input : \n{question} Let us think step by step.\n### Response :\n"
        # print(Twoshot)
        input_ids = tokenizer(nShots, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids = input_ids, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1,stopping_criteria=stopping_criteria)

        text = tokenizer.decode(outputs.sequences[0][len(input_ids[0]):].tolist(), clean_up_tokenization_spaces=True).strip()
        # print(text)
        # if(data[entry]['label'] not in text):
        #     print("ANSWER not matched")
        #     continue
        root_nodes = []
        adjacency_matrixs = []
        allNodes = []
        structures = tree_structure.split(')\n(')
        for structure in structures:
            root_node = parseTreeString(structure)
            adjacency_matrix = create_adjacency_matrix(root_node)
            allNode = get_nodes(root_node)
            root_nodes.append(root_node)
            adjacency_matrixs.append(adjacency_matrix)
            allNodes.append(allNode)
        tokenized_sentence, parent_words = tokenize_sentence_with_parent(data[entry]['question'], tokenizer)

        n = len(tokenized_sentence)
        matrix = [[0 for i in range(n)] for j in range(n)]
        for i in range(len(tokenized_sentence)):
            for j in range(i,len(tokenized_sentence)):
                if(i == j):
                    matrix[i][j] = 0
                for count in range(len(root_nodes)):
                    nodes = allNodes[count]
                    firstIndex = findIndexInMatrix(nodes, tokenized_sentence[i], parent_words[i])
                    secondIndex = findIndexInMatrix(nodes, tokenized_sentence[j], parent_words[j])
                    if(firstIndex != None and secondIndex != None):
                        if(adjacency_matrixs[count][firstIndex][secondIndex] == 1 or adjacency_matrixs[count][secondIndex][firstIndex] == 1):
                            # print(tokenized_sentence[i], tokenized_sentence[j])
                            matrix[i][j] = 1
                            matrix[j][i] = 1
        # print("matrix generated")
        allNodeEntities = []
        for nodes in allNodes:
            for node in nodes:
                allNodeEntities.append(node.name)


        # positivePair, negativePair, unrelatedPair = getPairs(tokenized_sentence, parent_words, matrix, allNodeEntities, typeA, typeB)

        # print(len(positivePair), len(negativePair), len(unrelatedPair))
        # breakpoint()
        start_index, _ = find_indices(tokenizer.tokenize(nShots), tokenizer.tokenize(data[entry]['question'])[2:])
        # breakpoint()
        print(start_index)
        if(start_index == None):
            continue
        hidden_layer_sim = {}
        for typeA in types:
            for typeB in types:
                hidden_layer_sim[f'{typeA}_{typeB}_positive'] = []
                hidden_layer_sim[f'{typeA}_{typeB}_negative'] = []
                hidden_layer_sim[f'{typeA}_{typeB}_unrelated'] = []
                # hidden_layer_sim[f'{typeA}_{typeB}_consecutive'] = []
                # hidden_layer_sim[f'{typeA}_{typeB}_random'] = []
        # hidden_layer_sim = {'positive':[], 'negative':[], 'unrelated':[]}
        twoShotTokenizer = tokenizer.tokenize(nShots)
        layer_count = 0
        for hidden_layer_mid in tqdm(outputs.hidden_states[0]):
            hidden_layer_mid = hidden_layer_mid.squeeze(0).cpu().detach()
            # breakpoint()
            umembed = model.lm_head
            layer_norm = model.model.norm
            resid_mid_logit = umembed(layer_norm(hidden_layer_mid.to(device)))
            probs = torch.softmax(resid_mid_logit[-1,:], dim=-1)
            top_indices = torch.topk(probs, k=1, dim=-1).indices
            top_indices_np = top_indices.cpu().numpy()
            # if(layer_count == 0 or layer_count == 1):
            #     print(f"layer_count : {layer_count}")
            #     print("Top 5 tokens from the hidden layer")
            #     for ind in top_indices_np:
            #         print(tokenizer.decode(ind), end='\n')
            #     print("\n")
            # layer_count += 1

            # top 5 tokens from the hidden layer
            # model.un
        #    resid_mid_logit = unembed(ln_final(hidden_layer_mid))
            # pca = PCA(n_components=64)
            # pca.fit(hidden_layer)
            # hidden_layer = pca.transform(hidden_layer)
            
            for typeA in types:
                for typeB in types:
                    positivePair, negativePair, unrelatedPair, consecutivePair, randomPair  = getPairs(tokenized_sentence, parent_words, matrix, allNodeEntities, typeA, typeB)
                    # print(len(positivePair), len(negativePair), len(unrelatedPair))
                    # postiveSimiliarity = 0
                    # negativeSimiliarity = 0
                    # unrelatedSimiliarity = 0
                    layerPositive = []
                    layerNegative = []
                    layerUnrelated = []
                    layerConsecutive = []
                    layerRandom = []
                    count = 0
                    for pair in positivePair:
                        if(twoShotTokenizer[start_index + pair[0]] == twoShotTokenizer[start_index + pair[1]]):
                            continue
                        # breakpoint()
                        postiveSimiliarity = [hidden_layer_mid[start_index + pair[0]].numpy(), hidden_layer_mid[start_index + pair[1]].numpy()]
                        layerPositive.extend([postiveSimiliarity])
                        # hidden_layer_sim[f'{typeA}_{typeB}_positive'].append(float(postiveSimiliarity))
                        count += 1

                    count = 0
                    for pair in negativePair:
                        if(twoShotTokenizer[start_index + pair[0]] == twoShotTokenizer[start_index + pair[1]]):
                            continue
                        negativeSimiliarity = [hidden_layer_mid[start_index + pair[0]].numpy(), hidden_layer_mid[start_index + pair[1]].numpy()]
                        layerNegative.extend([negativeSimiliarity])
                        # hidden_layer_sim[f'{typeA}_{typeB}_negative'].append(float(negativeSimiliarity))
                        count += 1

                    count = 0
                    for pair in unrelatedPair:
                        if(twoShotTokenizer[start_index + pair[0]] == twoShotTokenizer[start_index + pair[1]]):
                            continue
                        unrelatedSimiliarity = [hidden_layer_mid[start_index + pair[0]].numpy(), hidden_layer_mid[start_index + pair[1]].numpy()]
                        layerUnrelated.extend([unrelatedSimiliarity])
                    
                    # count = 0
                    # for pair in consecutivePair:
                    #     # if(twoShotTokenizer[start_index + pair[0]] == twoShotTokenizer[start_index + pair[1]]):
                    #     #     continue
                    #     consecutiveSimiliarity = cosine_similarity(hidden_layer[start_index + pair[0]], hidden_layer[start_index + pair[1]])
                    #     layerConsecutive.append(float(consecutiveSimiliarity))

                    # count = 0
                    # for pair in randomPair:
                    #     # if(twoShotTokenizer[start_index + pair[0]] == twoShotTokenizer[start_index + pair[1]]):
                    #     #     continue
                    #     randomSimiliarity = cosine_similarity(hidden_layer[start_index + pair[0]], hidden_layer[start_index + pair[1]])
                    #     layerRandom.append(float(randomSimiliarity))
                    # breakpoint()
                    hidden_layer_sim[f'{typeA}_{typeB}_positive'].append(torch.Tensor(layerPositive))
                    hidden_layer_sim[f'{typeA}_{typeB}_negative'].append(torch.Tensor(layerNegative))
                    hidden_layer_sim[f'{typeA}_{typeB}_unrelated'].append(torch.Tensor(layerUnrelated))
                    # hidden_layer_sim[f'{typeA}_{typeB}_consecutive'].append(layerConsecutive)
                    # hidden_layer_sim[f'{typeA}_{typeB}_random'].append(layerRandom)
            hidden_layer_mid.detach().cpu()
            del hidden_layer_mid
        


        allSimiliarity.append(hidden_layer_sim)
        print(len(model.residualMidOutput))
        model.residualMidOutput = []
        del outputs, input_ids
        torch.cuda.empty_cache()
        if(entry % 8 == 0):
            with open(f'{directory}/chunks/chunk_{chunk_number}.pickle', 'wb') as f:
                    pickle.dump(allSimiliarity, f)
            chunk_number += 1
            # allSimiliarity = []

            # items_per_chunk = 10

            # # Calculate the number of chunks
            # num_chunks = len(allSimiliarity) // items_per_chunk + (len(allSimiliarity) % items_per_chunk != 0)

            # # Save each chunk as a separate pickle file
            # for chunk_number in range(num_chunks):
            #     start_idx = chunk_number * items_per_chunk
            #     end_idx = (chunk_number + 1) * items_per_chunk
            #     chunk_data = allSimiliarity[start_idx:end_idx]

            #     with open(f'{directory}/chunks/chunk_{chunk_number}.pickle', 'wb') as f:
            #         pickle.dump(chunk_data, f)


                # json.dump(allSimiliarity, f)
        if(chunk_number == 26):
            break
    except Exception as e:
        print(e)
        print("ERROR")
        continue
# %%
# import pickle 

# items_per_chunk = 10

# Calculate the number of chunks
# num_chunks = len(allSimiliarity) // items_per_chunk + (len(allSimiliarity) % items_per_chunk != 0)

# Save each chunk as a separate pickle file
# for chunk_number in range(num_chunks):
    # start_idx = chunk_number * items_per_chunk
    # end_idx = (chunk_number + 1) * items_per_chunk
    # chunk_data = allSimiliarity[start_idx:end_idx]

    # with open(f'{directory}/chunks/chunk_{chunk_number}.pickle', 'wb') as f:
        # pickle.dump(chunk_data, f)


# with open(f'{directory}/pairsInfo_full.pickle', 'wb') as f:
        # pickle.dump(allSimiliarity, f)

# for typeA in types:
#     for typeB in types:
#         import os
#         os.makedirs(f'plots4/{typeA}_{typeB}', exist_ok=True)
#         import matplotlib.pyplot as plt

#         for i in range(len(allSimiliarity)):
#             plt.plot(allSimiliarity[i][f'{typeA}_{typeB}_positive'], label=f'question {i}')
#         # plt.legend()
#         # plt.show()
#         plt.savefig(f'plots4/{typeA}_{typeB}/similiarity_positive.png')
#         plt.clf()

#         for i in range(len(allSimiliarity)):
#             plt.plot(allSimiliarity[i][f'{typeA}_{typeB}_negative'], label=f'question {i}')
#         # plt.legend()
#         # plt.show()
#         plt.savefig(f'plots4/{typeA}_{typeB}/similiarity_negative.png')
#         plt.clf()

#         for i in range(len(allSimiliarity)):
#             plt.plot(allSimiliarity[i][f'{typeA}_{typeB}_unrelated'], label=f'question {i}')
#         # plt.legend()
#         # plt.show()
#         plt.savefig(f'plots4/{typeA}_{typeB}/similiarity_unrelated.png')
#         plt.clf()

#         # clear all plots
#         plt.cla()
#         plt.clf()
#         plt.close()



    
    

# %%



