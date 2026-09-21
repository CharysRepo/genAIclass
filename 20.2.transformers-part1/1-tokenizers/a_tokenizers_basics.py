from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer)

texts = [
    "how are you?",
    "heoll are you?",
    "Akwirw ier",
    "Hello, do you like tea? In the sunlit terraces of someunknownPlace.",
]

# individual tokenization
for text in texts:
    print(text)
    inputs = tokenizer.encode(text)  # convert text to tokens + tokens to ids
    print(inputs)
    output = tokenizer.decode(inputs)  # convert ids to tokens + grouping tokens
    print(output)

# batch tokenization  
batch = tokenizer(texts)
print(batch)
print(batch["input_ids"])
print(batch["attention_mask"])