from pydantic_ai import Agent

chat_agent = Agent(
    "ollama:llama3-groq-tool-use:8b", 
    instructions="You are helpful assistant."
)