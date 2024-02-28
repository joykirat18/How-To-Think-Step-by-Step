
import pickle
import torch
import io
from transformers import LlamaForCausalLM, LlamaTokenizer
MODEL_PATH = '/home/models//Llama-2-7b-hf'

tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)

noise_index = 0

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

#contents = pickle.load(f) becomes...

for noise_index in range(10):
    print("Noise Index: ", noise_index)
    try:
        
        with open(f'/home//StepByStep/results/reasoning/combined/{noise_index}/accuracy/Activation_example_start_0_end_6.pickle', 'rb') as f:
            start_0_end_6 = CPU_Unpickler(f).load()
        input_id_start_0_end_6 = start_0_end_6['input_ids']
        token_X_id = torch.argmax(start_0_end_6['patched_logits'][:, -1, :], dim=1)
        answer_token_start_0_end_6 = []
        for id in token_X_id:
            answer_token_start_0_end_6.append(tokenizer.decode(id.item()))
        print("start_0_end_6: ", answer_token_start_0_end_6)
        del start_0_end_6
    except Exception as error:
        print(f'{noise_index}, start_0_end_6 not found {error}' )
    
    try:
        with open(f'/home//StepByStep/results/reasoning/combined/{noise_index}/accuracy/Activation_example_start_5_end_16.pickle', 'rb') as f:
            start_5_end_16 = CPU_Unpickler(f).load()
        input_id_start_5_end_16 = start_5_end_16['input_ids']

        token_X_id = torch.argmax(start_5_end_16['patched_logits'][:, -1, :], dim=1)
        answer_token_start_5_end_16 = []
        for id in token_X_id:
            answer_token_start_5_end_16.append(tokenizer.decode(id.item()))
        print("start_5_end_16: ", answer_token_start_5_end_16)
        del start_5_end_16
    except Exception as error:
        print(f'{noise_index}, start_5_end_16 not found, {error}')

    try:
        with open(f'/home//StepByStep/results/reasoning/combined/{noise_index}/accuracy/Activation_example_start_15_end_26.pickle', 'rb') as f:
            start_15_end_26 = CPU_Unpickler(f).load()
        input_id_start_15_end_26 = start_15_end_26['input_ids']

        token_X_id = torch.argmax(start_15_end_26['patched_logits'][:, -1, :], dim=1)
        answer_token_start_15_end_26 = []
        for id in token_X_id:
            answer_token_start_15_end_26.append(tokenizer.decode(id.item()))
        print("start_15_end_26: ", answer_token_start_15_end_26)
        del start_15_end_26
    except Exception as error:
        print(f'{noise_index}, start_15_end_26 not found, {error}')
    
    try:
        with open(f'/home//StepByStep/results/reasoning/combined/{noise_index}/accuracy/Activation_example_start_25_end_36.pickle', 'rb') as f:
            start_25_end_36 = CPU_Unpickler(f).load()
        input_id_start_25_end_36 = start_25_end_36['input_ids']

        token_X_id = torch.argmax(start_25_end_36['patched_logits'][:, -1, :], dim=1)
        answer_token_start_25_end_36 = []
        for id in token_X_id:
            answer_token_start_25_end_36.append(tokenizer.decode(id.item()))
        print("start_25_end_36: ", answer_token_start_25_end_36)
        del start_25_end_36
    except Exception as error:
        print(f'{noise_index}, start_25_end_36 not found, {error}')
    
    try:
        with open(f'/home//StepByStep/results/reasoning/combined/{noise_index}/accuracy/Activation_example_start_35_end_46.pickle', 'rb') as f:
            start_35_end_46 = CPU_Unpickler(f).load()
        input_id_start_35_end_46 = start_35_end_46['input_ids']

        token_X_id = torch.argmax(start_35_end_46['patched_logits'][:, -1, :], dim=1)
        answer_token_start_35_end_46 = []
        for id in token_X_id:
            answer_token_start_35_end_46.append(tokenizer.decode(id.item()))
        print("start_35_end_46: ", answer_token_start_35_end_46)
        del start_35_end_46
    except Exception as error:
        print(f'{noise_index}, start_35_end_46 not found, {error}')

    data_input_id = []
    # iterate over all the files in the directory
    directory = f'/home//StepByStep/InformationFlow/result/{noise_index}'
    import os
    for filename in os.listdir(directory):
        if (filename.endswith(".pickle") and filename.startswith("example_")):
            # print(os.path.join(directory, filename))
            with open(os.path.join(directory, filename), 'rb') as f:
                data = pickle.load(f)
            number_filename = int(filename.split('_')[-1].split('.')[0])
            if(number_filename >= 0 and number_filename < 5):
                # print(number_filename)
                data_input_id.append({'data' : data, 'input_id' : input_id_start_0_end_6[number_filename], 'answer' : answer_token_start_0_end_6[number_filename]})
            elif(number_filename >= 5 and number_filename < 15):
                # print(number_filename)
                data_input_id.append({'data' : data, 'input_id' : input_id_start_5_end_16[number_filename - 5], 'answer' : answer_token_start_5_end_16[number_filename - 5]})
            elif(number_filename >= 15 and number_filename < 25):
                # print(number_filename)
                data_input_id.append({'data' : data, 'input_id' : input_id_start_15_end_26[number_filename - 15], 'answer' : answer_token_start_15_end_26[number_filename - 15]})
            elif(number_filename >= 25 and number_filename < 35):
                # print(number_filename)
                data_input_id.append({'data' : data, 'input_id' : input_id_start_25_end_36[number_filename - 25], 'answer' : answer_token_start_25_end_36[number_filename - 25]})
            elif(number_filename >= 35 and number_filename < 45):
                # print(number_filename)
                data_input_id.append({'data' : data, 'input_id' : input_id_start_35_end_46[number_filename - 35], 'answer' : answer_token_start_35_end_46[number_filename - 35]})
            else:
                print('error', number_filename)
                # if noise['start'] == 0 and noise['end'] == 6:
                #     input_id_start_0_end_6 = noise['input_ids']
                # elif noise['start'] == 5 and noise['end'] == 16:
                #     input_id_start_5_end_16 = noise['input_ids']
                # elif noise['start'] == 15 and noise['end'] == 26:
                #     input_id_start_15_end_26 = noise['input_ids']
                # else:
                #     print('error')
                # del noise

    with open(f'/home//StepByStep/InformationFlow/result/{noise_index}/data_input_id.pickle', 'wb') as f:
        pickle.dump(data_input_id, f)