from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print(tokenizer)

texts = [
    "how are you?",
    "Akwirw ier",
    "Hello, do you like tea? In the sunlit terraces of someunknownPlace.",
]
# padding to longest text
batch_dynamic = tokenizer(texts, padding=True, truncation=True)
for ids in batch_dynamic["input_ids"]:
    print(f"IDs   : {ids}")
    print(f"Decode: {tokenizer.decode(ids)}\n")
print("----------------------------")

# padding to max length
batch_max = tokenizer(texts, padding="max_length", truncation=True, max_length=10)

for ids in batch_max["input_ids"]:
    print(f"IDs   : {ids}")
    print(f"Decode: {tokenizer.decode(ids)}\n")

for ids in batch_max["input_ids"]:
    print(f"IDs   : {ids}")
    print(f"Decode: {tokenizer.decode(ids, skip_special_tokens=True)}\n")

# return overflow tokens beyond max length
batch_max = tokenizer(texts, padding="max_length", truncation=True, max_length=10, return_overflowing_tokens=True, return_length=True)
for ids, lengths in zip(batch_max["input_ids"], batch_max["length"]):
    print(f"IDs   : {ids}")
    print(f"Lenghts: {lengths}")
    print(f"Decode: {tokenizer.decode(ids, skip_special_tokens=True)}\n")