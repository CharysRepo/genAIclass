import torch
from transformers import AutoConfig, AutoTokenizer, pipeline, GPT2LMHeadModel
from torchinfo import summary

def get_inference_pipeline():
    model_id = "gpt2"
    context_length = 128
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    print(model)
    summary(model)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

def infer(pipe, user_input: str) -> str:
    outputs = pipe(
        user_input,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    return outputs[0]["generated_text"]

if __name__ == "__main__":
    gen_pipe = get_inference_pipeline()
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break
        result = infer(gen_pipe, user_input)
        print(result)

# Artificial intelligence is changing
# The scientific community
# Electric vehicles are transforming the global automotive industry by
# The transition to remote work completely changed corporate culture




