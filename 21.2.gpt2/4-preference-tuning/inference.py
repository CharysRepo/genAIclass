import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from trl import setup_chat_format
from pathlib import Path

def get_inference_pipeline(model_path: str):    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
    return pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        device=device
    )

def infer(pipe, user_input: str) -> str:
    chat_prompt = [{"role": "user", "content": user_input}]
    outputs = pipe(
        chat_prompt, 
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    return outputs[0]['generated_text'][-1]["content"]

if __name__ == "__main__":
    model_dir = Path(__file__).parent.parent.resolve() / "pt-model-512" / "checkpoint-708"
    gen_pipe = get_inference_pipeline(model_dir)

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break
        result = infer(gen_pipe, user_input)
        print(result)

# Describe the function of a computer motherboard
# Create a Python function to reverse a string. 
# Explain recursion to a 10-year-old.
# Give three tips for staying healthy.




