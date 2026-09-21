import os
os.environ["HF_HOME"] = "F:/hf"
from pathlib import Path
from multiprocessing import cpu_count
from datasets import load_dataset

def format_smollm2(sample):
    messages = []
    
    for conversation in sample["conversations"]:
        if conversation["from"] == "system":
            messages.append({"role": "system", "content": conversation["value"].strip()})
            
        elif conversation["from"] == "human":
            messages.append({"role": "user", "content": conversation["value"].strip()})
            
        elif conversation["from"] == "gpt":
            messages.append({"role": "assistant", "content": conversation["value"].strip()})
                
    return {"messages": messages} 


if __name__ == "__main__":    
    hermes_raw = load_dataset("NousResearch/hermes-function-calling-v1", split="train")
    print(hermes_raw)
    print(hermes_raw[0])

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