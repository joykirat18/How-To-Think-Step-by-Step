# %%
import json
from fewShotConstants import TemplateCOT_false, TemplateCOT_fictional, TemplateCOT_3Hop_fictional, TemplateCOT_3Hop_fictional_1step, TemplateCOT_3Hop_fictional_2step, TemplateCOT_3Hop_false, TemplateCOT_3Hop_false_1step, TemplateCOT_3Hop_false_2step, TemplateCOT_4Hop_false, TemplateCOT_4Hop_false_1step, TemplateCOT_4Hop_false_2step, TemplateCOT_4Hop_fictional, TemplateCOT_4Hop_fictional_1step, TemplateCOT_4Hop_fictional_2step
# Template = TemplateCOT_false


import argparse
parser = argparse.ArgumentParser()
# # python3 IndividualHeadAccuracy.py --noiseIndex 0 --device "cuda:1" --numExamples 10 --alteranteExamples 10
parser.add_argument("-filename", "--filename", help = "filename")
parser.add_argument("-device", "--device", help = "device")
parser.add_argument("-step", "--step", help = "step")
parser.add_argument("-example", "--example", help = "example")

# parser.add_argument("-numExamples", "--numExamples", help = "numExamples")
# parser.add_argument("-alternateExamples", "--alternateExamples", help = "alternateExamples")

# # alternateExamples
# parser.add_argument("--generate", action="store_true", help = "generate")
# # parser.add_argument("-modelPath", "--modelPath", help = "modelPath")
# parser.add_argument("--combined", action="store_true", help = "combined")
# # parser.add_argument("-combine_type", "--combine_type", help = "combine_type")

args = parser.parse_args()

filename = args.filename
steps = int(args.step)
device = args.device
example = int(args.example)
# 2hopData_without_distractors_2.json
# steps = 1
# device = 'cuda:2'
if('2hop' in filename and 'falseontology' in filename):
    hops = '2hop'
    kind = 'false'
    if(steps == 0):
        Template = TemplateCOT_false
    # elif(steps == 1):
        # Template = TemplateCOT_false_1step
elif('2hop' in filename):
    hops = '2hop'
    kind = 'fictional'
    if(steps == 0):
        Template = TemplateCOT_fictional
    # elif(steps == 1):
        # Template = TemplateCOT_fictional_1step
elif('3hop' in filename and 'falseontology' in filename):
    hops = '3hop'
    kind = 'false'
    if(steps == 0):
        Template = TemplateCOT_3Hop_false
    elif(steps == 1):
        Template = TemplateCOT_3Hop_false_1step
    elif(steps == 2):
        Template = TemplateCOT_3Hop_false_2step
elif('3hop' in filename):
    hops = '3hop'
    kind = 'fictional'
    if(steps == 0):
        Template = TemplateCOT_3Hop_fictional
    elif(steps == 1):
        Template = TemplateCOT_3Hop_fictional_1step
    elif(steps == 2):
        Template = TemplateCOT_3Hop_fictional_2step
elif('4hop' in filename and 'falseontology' in filename):
    hops = '4hop'
    kind = 'false'
    if(steps == 0):
        Template = TemplateCOT_4Hop_false
    elif(steps == 1):
        Template = TemplateCOT_4Hop_false_1step
    elif(steps == 2):
        Template = TemplateCOT_4Hop_false_2step
elif('4hop' in filename):
    hops = '4hop'
    kind = 'fictional'
    if(steps == 0):
        Template = TemplateCOT_4Hop_fictional
    elif(steps == 1):
        Template = TemplateCOT_4Hop_fictional_1step
    elif(steps == 2):
        Template = TemplateCOT_4Hop_fictional_2step
print(Template)
savefilename = f'data/noiseCOT_{kind}_{steps}_step_answer_{hops}_distractors_llama7b_Incorrect.json'
print(savefilename)
with open(filename, 'r') as f:
    COTData = json.load(f)
# %%
available_property_families = ["blue", "red", "brown", "orange",
							"small", "large",
							"metallic", "wooden", "luminous", "liquid",
							"transparent", "opaque",
							"nervous", "happy", "feisty", "shy",
							"bright", "dull",
							"sweet", "sour", "spicy", "bitter",
							"floral", "fruity", "earthy",
							"hot", "cold", "temperate",
							"kind", "mean", "angry", "amenable", "aggressive",
							"melodic", "muffled", "discordant", "loud",
							"slow", "moderate", "fast",
							"windy", "sunny", "overcast", "rainy", "snowy"]
available_entity_names = ["Fae", "Rex", "Sally", "Max", "Alex", "Sam", "Polly", "Stella", "Wren"]
if('falseontology' in filename):
    available_concept_names = ["animal", "vertebrate", "mammal", "carnivore", "feline", "cat", "dog", "sheep", "cow", "snake", "animal", "invertebrate", "arthropod", "insect", "lepidopteran", "butterfly", "moth", "ant", "spider", "crustacean", "Animal", "Vertebrate", "Mammal", "Carnivore", "Feline", "Cat", "Dog", "Sheep", "Cow", "Snake", "Animal", "Invertebrate", "Arthropod", "Insect", "Lepidopteran", "Butterfly", "Moth", "Ant", "Spider", "Crustacean"]
else:
    available_concept_names = ["wumpus", "yumpus", "zumpus", "dumpus", "rompus", "numpus", "tumpus", "vumpus", "impus", "jompus", "gorpus", "shumpus", "lempus", "sterpus", "grimpus", "lorpus", "brimpus","Wumpus", "Yumpus", "Zumpus", "Dumpus", "Rompus", "Numpus", "Tumpus", "Vumpus", "Impus", "Jompus", "Gorpus", "Shumpus", "Lempus", "Sterpus", "Grimpus", "Lorpus", "Brimpus"]

# def remove_every_and_capitalize(sentence):
#     # Define a regular expression pattern to match "Every" followed by a word
#     pattern = r'\bEvery\b (\w+)'

#     # Define a function to capitalize the matched word
#     def capitalize_word(match):
#         return match.group(1).capitalize()

#     # Use re.sub() with a callback function to process all occurrences
#     result = re.sub(pattern, capitalize_word, sentence)

#     return result

# def remove_each_and_capitalize(sentence):
#     # Define a regular expression pattern to match "Every" followed by a word
#     pattern = r'\bEach\b (\w+)'

#     # Define a function to capitalize the matched word
#     def capitalize_word(match):
#         return match.group(1).capitalize()

#     # Use re.sub() with a callback function to process all occurrences
#     result = re.sub(pattern, capitalize_word, sentence)

#     return result
# # %%
# filteredCOTData = []
# # available_property_families = ["blue", "red", "brown", "orange",
# # 							"small", "large",
# # 							"metallic", "wooden", "luminous", "liquid",
# # 							"transparent", "opaque",
# # 							"nervous", "happy", "feisty", "shy",
# # 							"bright", "dull",
# # 							"sweet", "sour", "spicy", "bitter",
# # 							"floral", "fruity", "earthy",
# # 							"hot", "cold", "temperate",
# # 							"kind", "mean", "angry", "amenable", "aggressive",
# # 							"melodic", "muffled", "discordant", "loud",
# # 							"slow", "moderate", "fast",
# # 							"windy", "sunny", "overcast", "rainy", "snowy"]
# for data in COTData:
#     old_property = ''
#     for property in available_property_families:
#         if('is not' in data['query'] and property in data['query']):
#             old_property = property
#             break
#     # print(old_property)
#     if(old_property != ''):
#         import random
#         available_properties_without_old = [prop for prop in available_property_families if prop != old_property]
#         new_property = random.choice(available_properties_without_old)
#         data['query'] = data['query'].replace(old_property, new_property)
#         data['query'] = data['query'].replace('is not', 'is')
#     if('not' in data['question'] or 'not' in data['answer']):
#         continue
#     if('Every' in data['question']):
#         data['question'] = remove_every_and_capitalize(data['question'])
#     if('Each' in data['question']):
#         data['question'] = remove_each_and_capitalize(data['question'])
#     if('Every' in data['answer']):
#         data['answer'] = remove_every_and_capitalize(data['answer'])
#     if('Each' in data['answer']):
#         data['answer'] = remove_each_and_capitalize(data['answer'])
    
#     else:
#         filteredCOTData.append({'question': data['question'], 'query': data['query'], 'answer': data['answer'], 'label' : data['label']})

#     # print(data)
#     split = data['answer'].split('.')
#     ans = ''
#     for i in range(len(split)):
#         if(i == 2):
#             continue
#         ans += split[i].strip() + '. '
#     ans = ans[:-2]

# # %%

# %%
# print(len(filteredCOTData))
# %%
from transformers import LlamaForCausalLM, LlamaTokenizer
import os

MODEL_PATH='/home/models/Llama-2-7b-hf'
print(MODEL_PATH)

tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
hf_model = LlamaForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True)
# %%

# %%
hf_model.to(device)

# %%
from transformers import pipeline, set_seed, StoppingCriteriaList, StoppingCriteria
import torch

# %%

# prompt += 'Alex is vumpus.'
# %%


def generate_text(prompt, max_new_tokens):

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    original_input_ids = input_ids.clone()
    # print(input_ids.shape)

    with torch.no_grad():
        logits = hf_model(input_ids).logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1)

    
    next_token = tokenizer.convert_ids_to_tokens(next_token_id)[0]

    intermedite_text_array = [('',next_token)]
# Generate the sentence one token at a time
    while max_new_tokens > 0:
        # Generate the next token's logits
        with torch.no_grad():
            logits = hf_model(input_ids).logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1)

        
        next_token = tokenizer.convert_ids_to_tokens(next_token_id)[0]
        # print(next_token)
        # break
        if(next_token == '<0x0A>'):
            break
        # Append the new token to the input_ids
        # print(next_token)
        if('▁' in next_token):
            # print("Next token is: " + next_token)
            intermedite_text_array.append((tokenizer.decode(input_ids[0][original_input_ids.size(1):], skip_special_tokens=True), next_token))
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(1)], dim=-1)
        max_new_tokens -= 1

    generated_text = tokenizer.decode(input_ids[0][original_input_ids.size(1):], skip_special_tokens=True)

    return generated_text, intermedite_text_array

# %%
def createNoiseData(prompt, intermediate_text):
    response = intermediate_text[0]
    next_token = intermediate_text[1].replace('▁', '')

    token_type = ''

    for concept in available_concept_names:
        if(concept.startswith(next_token)):
            token_type = 'concept'
            break

    for prop in available_property_families:
        if(prop.startswith(next_token)):
            token_type = 'property'
            break

    for entity in available_entity_names:
        if(entity.startswith(next_token)):
            token_type = 'entity'
            break
    # if(next_token == 'True' or next_token == 'False'):
        # token_type = 'bool'
    
    if(token_type == 'concept'):
        available_names = available_concept_names
    elif(token_type == 'property'):
        available_names = available_property_families
    elif(token_type == 'entity'):
        available_names = available_entity_names
    # elif(token_type == 'bool'):
        # available_names = available_property_families
    else:
        return None, None
        # print(next_token)
    
    splits = response.split(' ')
    for i in range(len(splits) - 1, -1, -1):

        if(splits[i].strip().startswith(next_token.strip()) and splits[i].strip() not in 'Let us think step by step.' and splits[i].strip() not in 'True or false:'):
            # print("found")
            old = splits[i]
            # if(token_type == 'bool'):
                # breakpoint()
            import random
            
            available_name_without_old = [prop for prop in available_names if prop != old]
            new = random.choice(available_name_without_old)
            if('.' in old):
                new += '.'
            splits[i] = new
            break
    noise_response = ' '.join(splits)

    # if(len(response) == 0 and len(noise_response) == 0):
        
    if(response.strip() == noise_response.strip()):
        splits = prompt.split(' ')
        # print(splits)
        for i in range(len(splits) - 1, -1, -1):
            if(splits[i].strip().startswith(next_token.strip()) and splits[i].strip() not in 'Let us think step by step.' and splits[i].strip() not in 'True or false:'):
                # print("found 2")
                old = splits[i]
                
                import random
                
                available_name_without_old = [prop for prop in available_names if prop != old]
                new = random.choice(available_name_without_old)
                if('.' in old):
                    new += '.'
                splits[i] = new
                break
        noise_prompt = ' '.join(splits)
    else:
        noise_prompt = prompt

    # print(noise_response + '|' + noise_prompt)
    # print(response + '|' + prompt)

    return noise_response, noise_prompt

# %%
# number = 0
# prompt = COTData[number]['question'] + ' ' + COTData[number]['query'] + ' Let us think step by step.' 
# prompt = Template.format(prompt)
noiseCOT = []
from tqdm import tqdm
accuracy = 0
pbar = tqdm(range(len(COTData)))
for number in pbar:
    try:
        prompt = COTData[number]['question'] + ' ' + COTData[number]['query'] + ' Let us think step by step.' 
        # prompt = Template.format(prompt)
        temp = {}
        temp['question'] = COTData[number]['question']
        temp['query'] = COTData[number]['query']
        if(steps == 0):
            answer = 'answer'
        elif(steps == 1):
            answer = '1-step'
        elif(steps == 2):
            answer = '2-step'
        temp[answer] = COTData[number][answer]
        temp['label'] = COTData[number]['label']
        temp['prompt'] = prompt


        generated_text, intermediate_array = generate_text(Template.format(prompt), 15)
        temp['generated'] = generated_text
        print(generated_text.strip()[:8])
        print(COTData[number][answer][:8])
        if(generated_text.strip()[:5] != COTData[number][answer][:5]):
            print("In Correct")
            accuracy += 1
            pbar.set_description(f"Accuracy: {accuracy/(number+1)}")
            pbar.refresh()
            count = 0
            # breakpoint()
            for i in range(len(intermediate_array)):
                noise_response, noise_prompt = createNoiseData(prompt, intermediate_array[i])
                
                if(noise_response == None):
                    continue
                temp[f'noise_response_{count}'] = noise_response
                temp[f'noise_prompt_{count}'] = noise_prompt
                temp[f'response_{count}'] = intermediate_array[i][0]
                count += 1
            noiseCOT.append(temp)
        with open(savefilename, 'w') as f:
            json.dump(noiseCOT, f)
        pbar.set_description(f"Accuracy: {accuracy/(number+1)}")
        pbar.refresh()
        print(len(noiseCOT))
        if(len(noiseCOT) == example):
            break
    except Exception as e:
        print(e)
        continue
    
    
    # break

# %%
# %%
with open(savefilename, 'w') as f:
    json.dump(noiseCOT, f)
# prompt = COTData[number]['question'] + ' ' + COTData[number]['query'] + ' Let us think step by step.' 
# response = generated_texts[index][0]
# next_token = generated_texts[index][1].replace('▁', '')

# token_type = ''

# for concept in available_concept_names:
#     if(concept.startswith(next_token)):
#         token_type = 'concept'
#         break

# for prop in available_property_families:
#     if(prop.startswith(next_token)):
#         token_type = 'property'
#         break

# for entity in available_entity_names:
#     if(entity.startswith(next_token)):
#         token_type = 'entity'
#         break
# # %%
# if(token_type == 'concept'):
#     available_names = available_concept_names
# elif(token_type == 'property'):
#     available_names = available_property_families
# elif(token_type == 'entity'):
#     available_names = available_entity_names
# else:
#     print("ERROR")
#     print(next_token)
# splits = response.split(' ')
# for i in range(len(splits) - 1, -1, -1):

#     if(splits[i].strip().startswith(next_token.strip())):
#         print("found")
#         old = splits[i]
        
#         import random
        
#         available_name_without_old = [prop for prop in available_names if prop != old]
#         new = random.choice(available_name_without_old)
#         if('.' in old):
#             new += '.'
#         splits[i] = new
#         break
# noise_response = ' '.join(splits)

# # if(len(response) == 0 and len(noise_response) == 0):
    
# if(response.strip() == noise_response.strip()):
#     splits = prompt.split(' ')
#     # print(splits)
#     for i in range(len(splits) - 1, -1, -1):
#         if(splits[i].strip().startswith(next_token.strip())):
#             print("found 2")
#             old = splits[i]
            
#             import random
            
#             available_name_without_old = [prop for prop in available_names if prop != old]
#             new = random.choice(available_name_without_old)
#             if('.' in old):
#                 new += '.'
#             splits[i] = new
#             break
#     noise_prompt = ' '.join(splits)
# else:
#     noise_prompt = prompt
# # %%
# print(noise_response + '|' + noise_prompt)
# print(response + '|' + prompt)


# %%

# %%
