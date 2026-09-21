from pathlib import Path
from datasets import load_dataset
from multiprocessing import cpu_count
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import setup_chat_format

def format_orca(example):
    prompt_chat = [{"role": "user", "content": example["question"]}]
    chosen_chat = prompt_chat + [{"role": "assistant", "content": example["chosen"]}]
    rejected_chat = prompt_chat + [{"role": "assistant", "content": example["rejected"]}]

    return {
        "prompt_chat": prompt_chat,
        "chosen_chat": chosen_chat,
        "rejected_chat": rejected_chat
    }

def tokenize_chat(batch, tokenizer, max_length):
    prompt_ids = tokenizer.apply_chat_template(batch["prompt_chat"], truncation=True,
        max_length=max_length, tokenize=True)
    chosen_input_ids = tokenizer.apply_chat_template(batch["chosen_chat"], truncation=True,
        max_length=max_length,tokenize=True)
    rejected_input_ids = tokenizer.apply_chat_template(batch["rejected_chat"], truncation=True, 
        max_length=max_length, tokenize=True,)
    
    # Automatically calculate the label shift
    # The length of 'prompt_ids' tells us exactly where the assistant response starts
    prompt_length = len(prompt_ids)
    
    # Construct labels: mask the prompt tokens with -100, keep the response tokens
    chosen_labels = [-100] * prompt_length + chosen_input_ids[prompt_length:]
    rejected_labels = [-100] * prompt_length + rejected_input_ids[prompt_length:]
    
    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": [1] * len(chosen_input_ids),
        "chosen_labels": chosen_labels,
        
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": [1] * len(rejected_input_ids),
        "rejected_labels": rejected_labels,
    }
    
if __name__ == "__main__":    
    orca_raw = load_dataset("Intel/orca_dpo_pairs", split="train")
    print(orca_raw)
    print(orca_raw[0])

    orca_cleaned = orca_raw.map(
        format_orca, 
        remove_columns=orca_raw.column_names, 
        num_proc=cpu_count()
    )
    print(orca_cleaned)
    print(orca_cleaned[0])

    orca_splitted = orca_cleaned.train_test_split(test_size=0.10)
    print(orca_splitted)
    print(orca_splitted["train"][0])

    tokenizer_dir = Path(__file__).parent.parent.resolve() / "tokenizer"
    model_dir = Path(__file__).parent.parent.resolve() / "foundation-model-512" / "checkpoint-14285"
    context_length = 512
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")
    
    tokenized_dataset = orca_splitted.map(
        tokenize_chat,
        fn_kwargs={
            "tokenizer": tokenizer, 
            "max_length": context_length
        },
        remove_columns=orca_splitted["train"].column_names, 
        num_proc=cpu_count()
    )
    print(tokenized_dataset)
    print(tokenized_dataset["train"][0])
    
    data_dir = Path(__file__).parent.resolve() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(data_dir)
