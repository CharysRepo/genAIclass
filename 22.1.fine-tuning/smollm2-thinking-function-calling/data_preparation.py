import os
os.environ["HF_HOME"] = "F:/hf"
from pathlib import Path
from multiprocessing import cpu_count
from datasets import load_dataset
from transformers import AutoTokenizer

def format_smollm2(sample):
    messages = []
    
    for conversation in sample["conversations"]:
        if conversation["role"] == "system":
            messages.append({"role": "system", "content": conversation["content"].strip()})
            
        elif conversation["role"] == "human":
            messages.append({"role": "user", "content": conversation["content"].strip()})
            
        elif conversation["role"] == "model":
            messages.append({"role": "assistant", "content": conversation["content"].strip()})

        elif conversation["role"] == "tool":
            messages.append({"role": "tool", "content": conversation["content"].strip()})
        
                
    return {"messages": messages} 

if __name__ == "__main__":    
    hermes_raw = load_dataset("Jofthomas/hermes-function-calling-thinking-V1", split="train")
    print(hermes_raw)

    hermes_dataset = hermes_raw.map(
        format_smollm2, 
        remove_columns=hermes_raw.column_names, 
        num_proc=cpu_count()
    )

    splitted_dataset = hermes_dataset.train_test_split(test_size=0.10)
    print(splitted_dataset)
    print(splitted_dataset["train"][0])
    
    data_dir = Path(__file__).parent.resolve() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    splitted_dataset.save_to_disk(data_dir)