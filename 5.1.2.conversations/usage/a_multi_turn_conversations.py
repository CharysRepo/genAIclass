from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'ollama:llama3-groq-tool-use:8b',
    instructions="You are helpful assistant."
)


def main():
    # multi-turn conversations
    print("### Multi turn Conversations\n")
    history = []
    while True:
        user_message = input(">>")
        if user_message == "q":
            break
        result = agent.run_sync(user_message, message_history=history)
        print(result.output)
        print(result.usage())
        history.extend(result.new_messages())

if __name__ == "__main__":
    main()

# Provide me with a good motto for today!
# Why did you choose this one?
# How about another one for my friend?
