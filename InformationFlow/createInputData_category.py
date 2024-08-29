
import pickle
import torch
import io
from transformers import LlamaForCausalLM, LlamaTokenizer
MODEL_PATH = 'meta-llama/Llama-2-7b-hf'

tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)

noise_index = 0

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

#contents = pickle.load(f) becomes...

for noise_index in range(5,6):
    print("Noise Index: ", noise_index)
    try:
        with open(f'result/reasoning/combined/{noise_index}/A/accuracy/Activation_example_start_0_end_10.pickle', 'rb') as f:
            start_0_end_10_A = CPU_Unpickler(f).load()
        input_id_start_0_end_10_A = start_0_end_10_A['input_ids']
        token_X_id = torch.argmax(start_0_end_10_A['patched_logits'][:, -1, :], dim=1)
        answer_token_start_0_end_10_A = []
        for id in token_X_id:
            answer_token_start_0_end_10_A.append(tokenizer.decode(id.item()))
        print("start_0_end_10_A: ", answer_token_start_0_end_10_A)
        del start_0_end_10_A
    except Exception as error:
        print(f'{noise_index}, start_0_end_10_A not found {error}' )
    
    try:
        with open(f'result/reasoning/combined/{noise_index}/A/accuracy/Activation_example_start_10_end_20.pickle', 'rb') as f:
            start_10_end_20_A = CPU_Unpickler(f).load()
        input_id_start_10_end_20_A = start_10_end_20_A['input_ids']
        token_X_id = torch.argmax(start_10_end_20_A['patched_logits'][:, -1, :], dim=1)
        answer_token_start_10_end_20_A = []
        for id in token_X_id:
            answer_token_start_10_end_20_A.append(tokenizer.decode(id.item()))
        print("start_10_end_20_A: ", answer_token_start_10_end_20_A)
        del start_10_end_20_A
    except Exception as error:
        print(f'{noise_index}, start_10_end_20_A not found {error}' )
        
    try:
        with open(f'result/reasoning/combined/{noise_index}/B/accuracy/Activation_example_start_0_end_10.pickle', 'rb') as f:
            start_0_end_10_B = CPU_Unpickler(f).load()
        input_id_start_0_end_10_B = start_0_end_10_B['input_ids']
        token_X_id = torch.argmax(start_0_end_10_B['patched_logits'][:, -1, :], dim=1)
        answer_token_start_0_end_10_B = []
        for id in token_X_id:
            answer_token_start_0_end_10_B.append(tokenizer.decode(id.item()))
        print("start_0_end_10_B: ", answer_token_start_0_end_10_B)                                                                                                                                                              
        del start_0_end_10_B
    except Exception as error:
        print(f'{noise_index}, start_0_end_10_B not found {error}' )
    
    try:
        with open(f'result/reasoning/combined/{noise_index}/B/accuracy/Activation_example_start_10_end_20.pickle', 'rb') as f:
            start_10_end_20_B = CPU_Unpickler(f).load()
        input_id_start_10_end_20_B = start_10_end_20_B['input_ids']
        token_X_id = torch.argmax(start_10_end_20_B['patched_logits'][:, -1, :], dim=1)
        answer_token_start_10_end_20_B = []
        for id in token_X_id:
            answer_token_start_10_end_20_B.append(tokenizer.decode(id.item()))
        print("start_10_end_20_B: ", answer_token_start_10_end_20_B)
        del start_10_end_20_B
    except Exception as error:
        print(f'{noise_index}, start_10_end_20_B not found {error}' )                                                                                                                                                                                                            
        
    try:                                                                                                                                                                                                                                                                                                                                                                            
        with open(f'result/reasoning/combined/{noise_index}/C/accuracy/Activation_example_start_0_end_10.pickle', 'rb') as f:
            start_0_end_10_C = CPU_Unpickler(f).load()
        input_id_start_0_end_10_C = start_0_end_10_C['input_ids']
        token_X_id = torch.argmax(start_0_end_10_C['patched_logits'][:, -1, :], dim=1)
        answer_token_start_0_end_10_C = []
        for id in token_X_id:
            answer_token_start_0_end_10_C.append(tokenizer.decode(id.item()))
        print("start_0_end_10_C: ", answer_token_start_0_end_10_C)
        del start_0_end_10_C
    except Exception as error:
        print(f'{noise_index}, start_0_end_10_C not found {error}' )
        
    try:
        with open(f'result/reasoning/combined/{noise_index}/C/accuracy/Activation_example_start_10_end_20.pickle', 'rb') as f:
            start_10_end_20_C = CPU_Unpickler(f).load()
            
        input_id_start_10_end_20_C = start_10_end_20_C['input_ids']
        token_X_id = torch.argmax(start_10_end_20_C['patched_logits'][:, -1, :], dim=1)
        answer_token_start_10_end_20_C = []
        for id in token_X_id:
            answer_token_start_10_end_20_C.append(tokenizer.decode(id.item()))
        print("start_10_end_20_C: ", answer_token_start_10_end_20_C)
        del start_10_end_20_C
    except Exception as error:
        print(f'{noise_index}, start_10_end_20_C not found {error}' )
        
    # try:
        # with open(f"")
        
    # try:
    #     with open(f'/home//StepByStep/results/reasoning/combined/{noise_index}/accuracy/Activation_example_start_5_end_16.pickle', 'rb') as f:
    #         start_5_end_16 = CPU_Unpickler(f).load()
    #     input_id_start_5_end_16 = start_5_end_16['input_ids']

    #     token_X_id = torch.argmax(start_5_end_16['patched_logits'][:, -1, :], dim=1)
    #     answer_token_start_5_end_16 = []
    #     for id in token_X_id:
    #         answer_token_start_5_end_16.append(tokenizer.decode(id.item()))
    #     print("start_5_end_16: ", answer_token_start_5_end_16)
    #     del start_5_end_16
    # except Exception as error:
    #     print(f'{noise_index}, start_5_end_16 not found, {error}')

    # try:
    #     with open(f'/home//StepByStep/results/reasoning/combined/{noise_index}/accuracy/Activation_example_start_15_end_26.pickle', 'rb') as f:
    #         start_15_end_26 = CPU_Unpickler(f).load()
    #     input_id_start_15_end_26 = start_15_end_26['input_ids']

    #     token_X_id = torch.argmax(start_15_end_26['patched_logits'][:, -1, :], dim=1)
    #     answer_token_start_15_end_26 = []
    #     for id in token_X_id:
    #         answer_token_start_15_end_26.append(tokenizer.decode(id.item()))
    #     print("start_15_end_26: ", answer_token_start_15_end_26)
    #     del start_15_end_26
    # except Exception as error:
    #     print(f'{noise_index}, start_15_end_26 not found, {error}')
    
    # try:
    #     with open(f'/home//StepByStep/results/reasoning/combined/{noise_index}/accuracy/Activation_example_start_25_end_36.pickle', 'rb') as f:
    #         start_25_end_36 = CPU_Unpickler(f).load()
    #     input_id_start_25_end_36 = start_25_end_36['input_ids']

    #     token_X_id = torch.argmax(start_25_end_36['patched_logits'][:, -1, :], dim=1)
    #     answer_token_start_25_end_36 = []
    #     for id in token_X_id:
    #         answer_token_start_25_end_36.append(tokenizer.decode(id.item()))
    #     print("start_25_end_36: ", answer_token_start_25_end_36)
    #     del start_25_end_36
    # except Exception as error:
    #     print(f'{noise_index}, start_25_end_36 not found, {error}')
    
    # try:
    #     with open(f'/home//StepByStep/results/reasoning/combined/{noise_index}/accuracy/Activation_example_start_35_end_46.pickle', 'rb') as f:
    #         start_35_end_46 = CPU_Unpickler(f).load()
    #     input_id_start_35_end_46 = start_35_end_46['input_ids']

    #     token_X_id = torch.argmax(start_35_end_46['patched_logits'][:, -1, :], dim=1)
    #     answer_token_start_35_end_46 = []
    #     for id in token_X_id:
    #         answer_token_start_35_end_46.append(tokenizer.decode(id.item()))
    #     print("start_35_end_46: ", answer_token_start_35_end_46)
    #     del start_35_end_46
    # except Exception as error:
    #     print(f'{noise_index}, start_35_end_46 not found, {error}')

    data_input_id = []
    # iterate over all the files in the directory
    directory = f'result/{noise_index}/A'
    import os
    for filename in os.listdir(directory):
        if (filename.endswith(".pickle") and filename.startswith("example_")):
            # print(os.path.join(directory, filename)) 
            with open(os.path.join(directory, filename), 'rb') as f:
                data = pickle.load(f)
            number_filename = int(filename.split('_')[-1].split('.')[0])
            if(number_filename >= 0 and number_filename < 10):
                data_input_id.append({'data' : data, 'input_id' : input_id_start_0_end_10_A[number_filename], 'answer' : answer_token_start_0_end_10_A[number_filename], 'category' : 'A'})
            elif(number_filename >= 10 and number_filename < 20):
                data_input_id.append({'data' : data, 'input_id' : input_id_start_10_end_20_A[number_filename-10], 'answer' : answer_token_start_10_end_20_A[number_filename-10], 'category' : 'A'})
            else:
                print('error', number_filename)
    
    directory = f'result/{noise_index}/B'
    for filename in os.listdir(directory):
        if (filename.endswith(".pickle") and filename.startswith("example_")):
            # print(os.path.join(directory, filename))
            with open(os.path.join(directory, filename), 'rb') as f:
                data = pickle.load(f)
            number_filename = int(filename.split('_')[-1].split('.')[0])
            if(number_filename >= 0 and number_filename < 10):
                data_input_id.append({'data' : data, 'input_id' : input_id_start_0_end_10_B[number_filename], 'answer' : answer_token_start_0_end_10_B[number_filename], 'category' : 'B'})
            elif(number_filename >= 10 and number_filename < 20):
                data_input_id.append({'data' : data, 'input_id' : input_id_start_10_end_20_B[number_filename-10], 'answer' : answer_token_start_10_end_20_B[number_filename-10], 'category' : 'B'})
            else:
                print('error', number_filename)
    
    directory = f'result/{noise_index}/C'
    for filename in os.listdir(directory):
        if (filename.endswith(".pickle") and filename.startswith("example_")):
            # print(os.path.join(directory, filename))
            with open(os.path.join(directory, filename), 'rb') as f:
                data = pickle.load(f)
            number_filename = int(filename.split('_')[-1].split('.')[0])
            if(number_filename >= 0 and number_filename < 10):
                data_input_id.append({'data' : data, 'input_id' : input_id_start_0_end_10_C[number_filename], 'answer' : answer_token_start_0_end_10_C[number_filename], 'category' : 'C'})
            elif(number_filename >= 10 and number_filename < 20):
                data_input_id.append({'data' : data, 'input_id' : input_id_start_10_end_20_C[number_filename-10], 'answer' : answer_token_start_10_end_20_C[number_filename-10], 'category' : 'C'})
            else:
                print('error', number_filename)

    with open(f'result/{noise_index}/data_input_id.pickle', 'wb') as f:
        pickle.dump(data_input_id, f)