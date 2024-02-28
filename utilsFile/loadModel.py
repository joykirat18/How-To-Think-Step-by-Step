
from transformer_lens import HookedTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import transformer_lens.utils as utils

def loadTransformerLensModel(modelPath):
    tokenizer = LlamaTokenizer.from_pretrained(modelPath)
    hf_model = LlamaForCausalLM.from_pretrained(modelPath, low_cpu_mem_usage=True)
    model = HookedTransformer.from_pretrained("meta-llama/Llama-2-7b-hf", hf_model=hf_model, device='cpu', fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer)

    return model, tokenizer

def loadTransformerModel(modelPath):
    tokenizer = LlamaTokenizer.from_pretrained(modelPath)
    hf_model = LlamaForCausalLM.from_pretrained(modelPath, low_cpu_mem_usage=True)

    return hf_model, tokenizer

def runCacheActivationPatching(model, input_ids):
    model.reset_hooks()
    fwd_hooks_list = []
    cache = {}
    def storeHookCache(value, hook):
        cache[hook.name] = torch.from_numpy(value.detach().cpu().numpy())
    for layer in range(model.cfg.n_layers):
        fwd_hooks_list.append((utils.get_act_name("z", layer, "attn"), storeHookCache))

    original_logits = model.run_with_hooks(input_ids, return_type="logits", fwd_hooks=fwd_hooks_list)

    return original_logits, cache

def runCacheCopying(model, input_ids):
    model.reset_hooks()
    fwd_hooks_list = []
    cache = {}
    def storeHookCache(value, hook):
        cache[hook.name] = torch.from_numpy(value.detach().cpu().numpy())
    for layer in range(model.cfg.n_layers):
        fwd_hooks_list.append((utils.get_act_name("z", layer), storeHookCache))
        fwd_hooks_list.append((utils.get_act_name("pattern", layer), storeHookCache))

    original_logits = model.run_with_hooks(input_ids, return_type="logits", fwd_hooks=fwd_hooks_list)

    return original_logits, cache