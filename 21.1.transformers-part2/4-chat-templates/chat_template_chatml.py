from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import setup_chat_format

messages = [
    {"role": "system", "content": "You are a helpful and concise AI assistant."},
    {"role": "user", "content": "How does a chat template work?"},
    {"role": "assistant", "content": "It formats chat turns into a single string with special tokens."}
]

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")
# it adds new tokens of chatml format to tokenizer
# it adjusts the vocabulary size of model
# it updates template of tokenizer

tokenized = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
print(tokenized)

tokenized = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=False
)
print(tokenized)

tokenized = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=False
)
print(tokenized)

tokenized = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=False, 
    return_tensors="pt"
)
print(tokenized)
