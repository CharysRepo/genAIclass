import asyncio
from pathlib import Path
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
import logfire
from dataclasses import dataclass

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()


@dataclass
class KnowledgeDeps:
    kb: str

kb_agent = Agent(
    #'groq:llama-3.3-70b-versatile',
     "ollama:glm-4.7-flash:q4_K_M",
    instructions="You are a helpful assistant."
)


@kb_agent.instructions
def get_knowledge_base(ctx: RunContext[KnowledgeDeps]) -> str:
    return f"Answer the questions based on the following Knowledge Base Content:\n{ctx.dp.kb}"

async def main():
    path = Path(__file__).parent / "data/small/kb.txt"
    deps = KnowledgeDeps(kb=path.read_text(encoding="utf-8"))

    message_history = []
    
    print("KB Agent is ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break

        try:
            result = await kb_agent.run(
                user_input, 
                deps=deps,
                message_history=message_history
            )
        except Exception as e:
            print(f"Error: {e}")
            continue
        message_history = result.all_messages()        
        print(f"Agent: {result.output}")
        print(result.usage())

if __name__ == "__main__":
    asyncio.run(main())


# How long does international shipping take?
# which Leather jackets are available in the store?
# get the contact details of Harvest & Mill store
# What is the return policy?
# RunUsage(input_tokens=1154, output_tokens=329, requests=1)