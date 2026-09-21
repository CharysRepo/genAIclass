from multiprocessing import cpu_count
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path 

def tokenize(element, tokenizer, max_length):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == max_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

if __name__ == '__main__':
    raw_dataset = load_dataset("Elriggs/openwebtext-100k", split="train")
    print(raw_dataset)

    splitted_dataset = raw_dataset.train_test_split(test_size=0.10)
    print(splitted_dataset)

    tokenizer_dir = Path(__file__).parent.parent.resolve() / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    context_length = 128
    tokenized_datasets = splitted_dataset.map(
         tokenize, 
         fn_kwargs={
            "tokenizer": tokenizer, 
            "max_length": context_length
        },
         batched=True, 
         remove_columns=splitted_dataset["train"].column_names, 
         num_proc=cpu_count()
    )
    print(tokenized_datasets) 
    print(f"Total tokens used for training: {len(tokenized_datasets["train"]) * context_length}")
    print(tokenized_datasets["train"][0])

    data_dir = Path(__file__).parent.resolve() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tokenized_datasets.save_to_disk(data_dir)
