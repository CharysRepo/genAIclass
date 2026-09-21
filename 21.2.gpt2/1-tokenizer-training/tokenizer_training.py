from datasets import load_dataset
from pathlib import Path
from transformers import AutoTokenizer

# Skylion007/openwebtext
ds_web = load_dataset("Elriggs/openwebtext-100k", split="train", streaming=True)
def batch_iterator(batch_size=1000):
    batch = []
    for example in ds_web:
        batch.append(example["text"])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=len(tokenizer))

tokenizer_dir = Path(__file__).parent.parent.resolve() / "tokenizer"
new_tokenizer.save_pretrained(tokenizer_dir)

