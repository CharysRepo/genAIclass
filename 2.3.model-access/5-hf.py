from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.providers.huggingface import HuggingFaceProvider
import os

load_dotenv(override=True)

model = HuggingFaceModel('Qwen/Qwen3.5-35B-A3B', provider=HuggingFaceProvider(api_key=os.getenv('HF_TOKEN'), provider_name='novita'))

agent = Agent(
    model,
    instructions="You are a helpful assistant.",
)

response = agent.run_sync("Write a haiku about recursion in programming.")
print(response.output)
print(response.usage())

response = agent.run_sync("What is recursion in programming.")
print(response.output)
print(response.usage())