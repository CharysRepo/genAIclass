import torch
from pathlib import Path
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

def get_inference_pipeline(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

def infer(pipe, start_phrase: str) -> str:
    outputs = pipe(
        start_phrase,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.1
    )
    return outputs[0]["generated_text"]

if __name__ == "__main__":
    #model_dir = Path(__file__).parent.parent.resolve() / "foundation-model-512" / "checkpoint-14285"
    model_dir = "Algorithmica/gpt2-foundation-model"
    gen_pipe = get_inference_pipeline(model_dir)

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break
        result = infer(gen_pipe, user_input)
        print(result)

