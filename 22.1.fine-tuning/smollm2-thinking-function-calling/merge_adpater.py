import os
os.environ["HF_HOME"] = "F:/hf"
from pathlib import Path
from torchinfo import summary
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_adapter_with_base(adapter_dir, output_dir):    
    model_dir = "HuggingFaceTB/SmolLM2-1.7B-Instruct"    
    base_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    print(peft_model)
    summary(peft_model)
    merged_model = peft_model.merge_and_unload()
    print(merged_model)
    summary(merged_model)

    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Successfully saved merged model and tokenizer to {output_dir}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    adapter_dir = base_dir / "smollm2-1.7b-it-thinking-function-calling-adapter-qlora" / "checkpoint-15"
    output_dir = base_dir / "smollm2-1.7b-thinking"
    merge_adapter_with_base(adapter_dir, output_dir)

