from dotenv import load_dotenv
from groq import AsyncClient
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.providers.huggingface import HuggingFaceProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
import os

load_dotenv(override=True)

model = GroqModel('llama-3.3-70b-versatile')
#model = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key=os.getenv('GROQ_API_KEY')))
# custom_http_client = AsyncClient(timeout=30)
# model = GroqModel(
#     'llama-3.3-70b-versatile',
#     provider=GroqProvider(api_key=os.getenv('GROQ_API_KEY'), http_client=custom_http_client),
# )
#model = HuggingFaceModel('Qwen/Qwen3.5-35B-A3B', provider=HuggingFaceProvider(api_key=os.getenv('HF_TOKEN'), provider_name='novita'))
# ollama_model = OpenAIChatModel(
#     'llama3-groq-tool-use:8b',
#     provider=OllamaProvider(base_url='http://localhost:11434/v1'),  
# )
agent = Agent(
    model,
    instructions="You are a helpful assistant.",
)

response = agent.run_sync("Write a haiku about recursion in programming.")
print(response.output)
print(response.usage())

response = agent.run_sync("list top 3 generative models for reasoning")
print(response.output)
print(response.usage())
