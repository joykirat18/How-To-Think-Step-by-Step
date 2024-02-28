from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast
import torch
import sys
import os
from peft import PeftModel, PeftConfig
from peft import LoraConfig, TaskType, get_peft_model
import sys
sys.path.append('../../JS')
from honest_llama import llama
def loadLoraCheckPoint(path_to_model, path_to_tokenizer, path_to_state_dict):
    tokenizer = llama.LLaMATokenizer.from_pretrained(path_to_tokenizer)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        )
    print("Loading lora checkpoint")
    model = AutoModelForCausalLM.from_pretrained(path_to_model, return_dict=True, low_cpu_mem_usage=True)
    model = get_peft_model(model, peft_config)
    # print(model)
    # model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(path_to_state_dict))

    return model, tokenizer

fineTunedModel = True
path_to_model = "/home//Project2/vicuna-7b/"  # push to runtime argument
path_to_tokenizer = "/home//Project2/vicuna-7b/" # push to runtime argument
# ../frugal_lms/saved_model/base_gptj/
device = 'cuda:0' # push it to runtime argument
# mt = ModelAndTokenizer(path_to_model, device=device)
# tokenizer = LlamaTokenizerFast.from_pretrained('hf-internal-testing/llama-tokenizer', is_fast=True)
# model = AutoModelForCausalLM.from_pretrained(path_to_model, low_cpu_mem_usage=True).to(device)
if(fineTunedModel):
    model, tokenizer = loadLoraCheckPoint(path_to_model, path_to_tokenizer, '/home//frugal_lms/FinalCodeBase/prontoqa/stateDict.pth')
    model.to(device)
else:
    tokenizer = llama.LLaMATokenizer.from_pretrained(path_to_tokenizer)
    model = llama.LLaMAForCausalLM.from_pretrained(path_to_model, low_cpu_mem_usage=True).to(device)

numberOfShots = 0
directory = "vicunaFineTuned/0shot_alphabetOntology"
os.makedirs(directory, exist_ok=True)
if(fineTunedModel):
    import json
    with open('/home//frugal_lms/FinalCodeBase/data/prontoqa_finetune/alphabetOntology.json', 'r') as f:
        data = json.load(f)
else:
    import json
    with open('data/2hop_250_QA_pair_distractor.json', 'r') as f:
        data = json.load(f)

shots = ""

with open('data/fewShotPrompts.json', 'r') as f:
    fewShots = json.load(f)


for i in range(numberOfShots):
    question = fewShots[i]['question'] + ' ' + fewShots[i]['query']
    shots += f"### Input : \n{question} Let us think step by step.\n### Response : \n{fewShots[i]['answer']}\n\n"

print(shots)
# import json
# with open('data/2hop_250_QA_pair_distractor.json', 'r') as f:
#     data = json.load(f)

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
        second_last_token = self.tokenizer.decode(input_ids[0][-2].tolist(), clean_up_tokenization_spaces=True)
        # print(last_token)
        if(fineTunedModel):
            if(last_token == '<|endoftext|>'):
                return True
        if(last_token == '<0x0A>'):
            return True
        return False
    
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer=tokenizer,stops=['stop_words'])])

output_ans = []

from tqdm import tqdm
for i in tqdm(range(250)):
    question = data[i]['question'] + ' ' + data[i]['query']
    tree_structure = data[i]['tree']
    if(fineTunedModel):
        nShots = f'{question} Let us think step by step'
    else:
        nShots = shots + f"### Input : \n{question} Let us think step by step.\n### Response :\n"
    # Twoshot = f"### Input : \nEvery phorpus is a twimpus. Phorpuses are kergits. Every phorpus is wooden. Each jempor is a phorpus. Jempors are borpins. Each jempor is happy. Every kergit is brown. Each borpin is not bright. Jempuses are not wooden. Every jelgit is a storpist. Jelgits are opaque. Rex is a jelgit. Rex is a jempor. True or false: Rex is wooden. Let us think step by step.\n### Response : \nRex is a jempor. Each jempor is a phorpus. Rex is a phorpus. Every phorpus is wooden. Rex is wooden. True\n\n### Input : \nBorpins are daumpins. Each borpin is a jelgit. Every borpin is not wooden. Phorpuses are wooden. Jempors are borpins. Every jempor is a jempus. Jempors are purple. Jelgits are transparent. Every jempus is happy. Every storpist is a twimpus. Storpists are not small. Sam is a storpist. Sam is a jempor. True or false: Sam is wooden. Let us think step by step.\n### Response : \nSam is a jempor. Jempors are borpins. Sam is a borpin. Every borpin is not wooden. Sam is not wooden. False\n\n### Input : \n{question} Let us think step by step.\n### Response :\n"
    # Twoshot = f"### Input : \n{question} Let us think step by step.\n### Response :\n"
    # print(Twoshot)
    input_ids = tokenizer(nShots, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids=input_ids, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=60,stopping_criteria=stopping_criteria)
    if(fineTunedModel):
        text = tokenizer.decode(outputs.sequences[0][len(input_ids[0]):].tolist(), clean_up_tokenization_spaces=True).strip()[5:]
        if('<|endoftext|>' in text):
            text = text[:text.find('<|endoftext|>')].strip()
    else:
        text = tokenizer.decode(outputs.sequences[0][len(input_ids[0]):].tolist(), clean_up_tokenization_spaces=True).strip()
    print(text)
    temp = data[i]
    temp['output'] = text
    output_ans.append(temp)

    import json
    with open(f'{directory}/2hop_QA_pair_distractor_output.json', 'w') as f:
        json.dump(output_ans, f)

correct_ans = []
for i in range(len(output_ans)):
    # predicted_label = None
    # ground_label = None
    # if('true' in output_ans[i]['output'].lower()):
        # predicted_label = 'true'
    # elif('false' in output_ans[i]['output'].lower()):
        # predicted_label = 'false'
    # ground_label = output_ans[i]['label']
    if(output_ans[i]['output'].lower().strip() == output_ans[i]['answer'].lower().strip()):
        correct_ans.append(output_ans[i])

print("Correct output: ", len(correct_ans))


with open(f'{directory}/2hop_QA_pair_distractor_output_correct.json', 'w') as f:
    json.dump(correct_ans, f)
