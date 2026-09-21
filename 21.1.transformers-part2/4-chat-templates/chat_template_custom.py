from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import setup_chat_format

messages = [
    {"role": "system", "content": "You are a helpful and concise AI assistant."},
    {"role": "user", "content": "How does a chat template work?"},
    {"role": "assistant", "content": "It formats chat turns into a single string with special tokens."}
]

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
custom_special_tokens = [
    "<system>", "</system>",
    "<user>", "</user>",
    "<assistant>", "</assistant>"
]

tokenizer.add_special_tokens({"additional_special_tokens": custom_special_tokens})
model.resize_token_embeddings(len(tokenizer))
# ninja template format
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{ '<' + message['role'] + '>' }}"
    "{{ message['content'] }}"
    "{{ '</' + message['role'] + '>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<assistant>' }}"
    "{% endif %}"
)

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
