import pickle
from pathlib import Path
import torch
import torch.nn.functional as F

def inference(model, tokenizer, start_phrase, max_gen_length, device):
    model.eval() 
    context_length = model.config["context_length"]
    
    generated_tokens = tokenizer.encode(start_phrase)    
    for _ in range(max_gen_length):
        context_tokens = generated_tokens[-context_length:] 
        input_tensor = torch.tensor([context_tokens], dtype=torch.long).to(device) 
 
        with torch.no_grad():
            logits = model(input_tensor)
        predicted_idx = torch.argmax(logits, dim=-1).item() 
        generated_tokens.append(predicted_idx)
        
    return tokenizer.decode(generated_tokens)

def inference_creative_with_topk_sampling(model, tokenizer, start_phrase, max_gen_length, device, temp, top_k):
    model.eval() 
    context_length = model.config["context_length"]
    
    generated_tokens = tokenizer.encode(start_phrase)    
    for _ in range(max_gen_length):
        context_tokens = generated_tokens[-context_length:] 
        input_tensor = torch.tensor([context_tokens], dtype=torch.long).to(device) 
 
        with torch.no_grad():
            logits = model(input_tensor)
        logits = logits / temp

        #handle the case when top_k value is beyond the vocabulary size
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        predicted_idx  = torch.multinomial(probs, num_samples=1)
        generated_tokens.append(predicted_idx.item())
        
    return tokenizer.decode(generated_tokens)

def inference_creative_with_topp_sampling(model, tokenizer, start_phrase, max_gen_length, device, temp, top_p):
    model.eval() 
    context_length = model.config["context_length"]
    
    generated_tokens = tokenizer.encode(start_phrase)    
    for _ in range(max_gen_length):
        context_tokens = generated_tokens[-context_length:] 
        input_tensor = torch.tensor([context_tokens], dtype=torch.long).to(device) 
 
        with torch.no_grad():
            logits = model(input_tensor)
            
        logits = logits / temp
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False  
        sorted_logits[sorted_indices_to_remove] = -float("Inf")

        sorted_probs = F.softmax(sorted_logits, dim=-1)
        sorted_predicted_idx = torch.multinomial(sorted_probs, num_samples=1)
        predicted_idx = sorted_indices.gather(dim=-1, index=sorted_predicted_idx)            
        generated_tokens.append(predicted_idx.item())
        
    return tokenizer.decode(generated_tokens)


def inference_creative_with_hybrid_sampling(model, tokenizer, start_phrase, max_gen_length, device, temp, top_k, top_p):
    model.eval() 
    context_length = model.config["context_length"]
    
    generated_tokens = tokenizer.encode(start_phrase)    
    for _ in range(max_gen_length):
        context_tokens = generated_tokens[-context_length:] 
        input_tensor = torch.tensor([context_tokens], dtype=torch.long).to(device) 
 
        with torch.no_grad():
            logits = model(input_tensor)            
        logits = logits / temp

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        actual_top_k = min(top_k, sorted_logits.size(-1))
        sorted_logits = sorted_logits[..., :actual_top_k]
        sorted_indices = sorted_indices[..., :actual_top_k]
        
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False  
        sorted_logits[sorted_indices_to_remove] = -float("Inf")

        sorted_probs = F.softmax(sorted_logits, dim=-1)
        sorted_predicted_idx = torch.multinomial(sorted_probs, num_samples=1)
        predicted_idx = sorted_indices.gather(dim=-1, index=sorted_predicted_idx)            
        generated_tokens.append(predicted_idx.item())
        
    return tokenizer.decode(generated_tokens)

def inference_creative_with_hybrid_sampling_and_rep_penalty(model, tokenizer, start_phrase, max_gen_length, device, temp, top_k, top_p, repetition_penalty=1.2):
    model.eval() 
    context_length = model.config["context_length"]
    
    generated_tokens = tokenizer.encode(start_phrase)    
    for _ in range(max_gen_length):
        context_tokens = generated_tokens[-context_length:] 
        input_tensor = torch.tensor([context_tokens], dtype=torch.long).to(device) 
 
        with torch.no_grad():
            logits = model(input_tensor)

        # apply temperature            
        logits = logits / temp

        # apply repetition penalty
        unique_tokens = list(set(context_tokens))
        prev_logits = logits[0, unique_tokens]        
        # If logit is positive, divide to reduce it. If negative, multiply to lower it further.
        logits[0, unique_tokens] = torch.where(
            prev_logits > 0, 
            prev_logits / repetition_penalty, 
            prev_logits * repetition_penalty
        )

        # apply top-k filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        actual_top_k = min(top_k, sorted_logits.size(-1))
        sorted_logits = sorted_logits[..., :actual_top_k]
        sorted_indices = sorted_indices[..., :actual_top_k]
        
        # apply top-p filtering
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False  
        sorted_logits[sorted_indices_to_remove] = -float("Inf")

        # sample from filtered sorted distribution
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        sorted_predicted_idx = torch.multinomial(sorted_probs, num_samples=1)
        predicted_idx = sorted_indices.gather(dim=-1, index=sorted_predicted_idx)            
        generated_tokens.append(predicted_idx.item())
        
    return tokenizer.decode(generated_tokens)


if __name__ == "__main__":
    model_dir = Path(__file__).resolve().parent / "models"
    model_path = model_dir / "generative_model_v1.pt"
    tokenizer_path = model_dir / "tokenizer.pkl"

    print("Loading tokenizer...")
    with open(tokenizer_path, 'rb') as f_tok:
        tokenizer = pickle.load(f_tok)

    print("Loading model...")
    model = torch.load(model_path, weights_only=False) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    while True:
        start_text = input(">>")
        if start_text in ["exit", "quit"]:
            break
        #output_text = inference(model, tokenizer, start_phrase=start_text, max_gen_length=10, device=device)
        #output_text = inference_creative_with_topk_sampling(model, tokenizer, start_phrase=start_text, max_gen_length=10, device=device, temp=0.7, top_k=10)
        #output_text = inference_creative_with_topp_sampling(model, tokenizer, start_phrase=start_text, max_gen_length=10, device=device, temp=0.7, top_p=0.8)
        #output_text = inference_creative_with_hybrid_sampling(model, tokenizer, start_phrase=start_text, max_gen_length=10, device=device, temp=0.7, top_k=10, top_p=0.8)
        output_text = inference_creative_with_hybrid_sampling_and_rep_penalty(model, tokenizer, start_phrase=start_text, max_gen_length=10, device=device, temp=0.7, top_k=10, top_p=0.8)

        print("Generated Text:", output_text)

"""
First Citizen : Before we
Second Citizen : One word ,
MENENIUS : Hail , honorable
BRUTUS : We do it
SICINIUS : The people are
"""

"""
resolved rather to die than
To make him worthy whose
goodly house and the buildings
they are in a most
"""