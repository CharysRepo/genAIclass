import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

def infer1(model, tokenizer, prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(prompt,return_tensors='pt').input_ids.to(device)
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, temperature=0.7, top_k=10, top_p=0.95)
    output = tokenizer.decode(outputs, skip_special_tokens=True)
    return output[0]


def infer2(model, tokenizer, prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device)
    outputs = pipe(
        prompt,
        max_new_tokens = 100,
        do_sample = True, 
        temperature = 0.7,
        top_k=10, 
        top_p=0.95
    )
    return outputs[0]["generated_text"]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print(infer1(model, tokenizer, "I was telling her that"))
    print(infer2(model, tokenizer, "I was telling her that"))
    #print(infer2(model, tokenizer, "write a python code to reverse string"))


