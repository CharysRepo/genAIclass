from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

texts = [
    "how are you?",
    "Akwirw ier",
    "Hello, do you like tea? In the sunlit terraces of someunknownPlace. got it. thanks!",
]
def batch_iterator(batch_size=1):
    batch = []
    for example in texts:
        batch.append(example["text"])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=len(tokenizer))
print(new_tokenizer)