from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv(override=True)

agent = Agent(
    #'ollama:llama3-groq-tool-use:8b',
    'groq:llama-3.3-70b-versatile',
    #"github:deepseek/DeepSeek-R1",
    #'openrouter:deepseek/deepseek-v3.2',
    #'openai:gpt-4o-mini',  
    #'google-gla:gemini-3-pro-preview',
    instructions="You are a helpful assistant.",
)

response = agent.run_sync("Write a haiku about recursion in programming.")
print(response.output)
print(response.usage())

response = agent.run_sync("list top 3 generative models for reasoning")
print(response.output)
print(response.usage())