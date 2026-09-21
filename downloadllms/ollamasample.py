from ollama  import chat

response = chat(
    model='tinyllama',
    messages=[{'role': 'user', 'content': 'Hello!'}],
)
print(response.message.content)