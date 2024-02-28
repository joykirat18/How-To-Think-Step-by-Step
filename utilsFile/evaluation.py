import torch.nn.functional as F
import torch
from scipy.stats import wasserstein_distance

def kl_divergence(input_logit, target_logit):
    input_next_token_logit = input_logit[:, -1, :]
    target_next_token_logit = target_logit[:, -1, :]

    input_logProb = F.log_softmax(input_next_token_logit, dim=-1)
    target_logProb = F.log_softmax(target_next_token_logit, dim=-1)

    kl_div = F.kl_div(input_logProb, target_logProb,log_target=True, reduction="none").sum(dim=-1)
    return kl_div.mean().detach().cpu()

def kl_divergence_mean_var(input_logit, target_logit):
    input_next_token_logit = input_logit[:, -1, :]
    target_next_token_logit = target_logit[:, -1, :]

    input_logProb = F.log_softmax(input_next_token_logit, dim=-1)
    target_logProb = F.log_softmax(target_next_token_logit, dim=-1)

    kl_div = F.kl_div(input_logProb, target_logProb,log_target=True, reduction="none").sum(dim=-1)
    return (kl_div.mean().detach().cpu(), kl_div.var().detach().cpu())

def wasserstein_dis(input_logit, target_logit):
    input_next_token_logit = input_logit[:, -1, :]
    target_next_token_logit = target_logit[:, -1, :]

    input_prob = F.softmax(input_next_token_logit, dim=-1)
    target_prob = F.softmax(target_next_token_logit, dim=-1)

    input_distribution = input_prob.detach().cpu().numpy()
    target_distribution = target_prob.detach().cpu().numpy()

    # Compute Wasserstein distance using pyemd library
    wasserstein_dist = wasserstein_distance(input_distribution.flatten(), target_distribution.flatten())
    
    return wasserstein_dist

def logit_accuracy(model, logits, labels):
    next_token_logits = logits[:, -1, :]
    next_token_ids = torch.argmax(next_token_logits, dim=-1)
    next_token_ids = [[tokens.item()] for tokens in next_token_ids]
    accuracy = 0
    next_tokens = model.tokenizer.batch_decode(next_token_ids)
    print(next_tokens)
    for i in range(len(next_tokens)):
        if(labels[i].strip().startswith(next_tokens[i].strip())):
            accuracy += 1
    return accuracy / len(labels)

def model_accuracy(model, input_ids, labels):
    model_accuracy = 0
    N = len(input_ids)
    for i in range(N):
        id = input_ids[i]
        with torch.no_grad():
            outputs = model(id)
            next_token_logits = outputs[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        next_token = model.tokenizer.decode(next_token_id).strip()
        # print(next_token, labels[i])
        if(labels[i].strip().startswith(next_token.strip())):
            model_accuracy += 1
        # if len(next_token) != 1 and next_token in dataset[i]['label']:
            # model_accuracy += 1
    return model_accuracy/N