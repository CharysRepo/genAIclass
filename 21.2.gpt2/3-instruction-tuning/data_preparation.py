from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from multiprocessing import cpu_count
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import setup_chat_format

def format_alpaca(example):
    """Formats Alpaca fields into standard chat messages."""
    user_input = example["instruction"]
    if example.get("input"):
        user_input += "\n" + example["input"]

    return {
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": example["output"]},
        ]
    }

def format_dolly(example):
    """Formats Dolly fields into standard chat messages."""
    user_input = example["instruction"]
    if example.get("context") and example["context"].strip():
        user_input += "\n" + example["context"]

    return {
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": example["response"]},
        ]
    }

def tokenize_chat(batch, tokenizer, max_length):
    tokenized = tokenizer.apply_chat_template(
        batch["messages"],
        truncation=True,
        max_length=max_length,
        add_generation_prompt=False,
        tokenize=True
    )
    return tokenized
    
if __name__ == "__main__":    
    alpaca_raw = load_dataset("tatsu-lab/alpaca", split="train")
    dolly_raw = load_dataset("databricks/databricks-dolly-15k", split="train")

    alpaca_clean = alpaca_raw.map(
        format_alpaca, 
        remove_columns=alpaca_raw.column_names, 
        num_proc=cpu_count()
    )
    dolly_clean = dolly_raw.map(
        format_dolly, 
        remove_columns=dolly_raw.column_names, 
        num_proc=cpu_count()
    )

    combined_dataset = concatenate_datasets([alpaca_clean, dolly_clean])
    combined_dataset = combined_dataset.shuffle(seed=1)
    splitted_dataset = combined_dataset.train_test_split(test_size=0.10)
    print(splitted_dataset)
    print(splitted_dataset["train"][0])

    tokenizer_dir = Path(__file__).parent.parent.resolve() / "tokenizer"
    model_dir = Path(__file__).parent.parent.resolve() / "foundation-model-512" / "checkpoint-14285"
    context_length = 512
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")
    
    tokenized_dataset = splitted_dataset.map(
        tokenize_chat,
        fn_kwargs={
            "tokenizer": tokenizer, 
            "max_length": context_length
        },
        remove_columns=splitted_dataset["train"].column_names, 
        num_proc=cpu_count()
    )
    print(tokenized_dataset)
    print(tokenized_dataset["train"][0])
    
    data_dir = Path(__file__).parent.resolve() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(data_dir)
