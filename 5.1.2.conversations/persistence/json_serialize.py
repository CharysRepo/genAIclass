from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent, ModelMessagesTypeAdapter
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'ollama:llama3-groq-tool-use:8b',
    instructions="You are helpful assistant."
)


def main():
    history = []
    while True:
        user_message = input(">>")
        if user_message == "q":
            break
        result = agent.run_sync(user_message, message_history=history)
        print(result.output)
        history.extend(result.new_messages())

    # summarization
    summary_prompt = (
        "Please summarize the whole conversation until this message. "
        "Point out key topics and provide a timeline of events from this conversation."
    )
    result = agent.run_sync(summary_prompt, message_history=history)
    print(result.output)

    # persist history
    history_json = ModelMessagesTypeAdapter.dump_json(result.all_messages()) # serialize
    save_path = Path(__file__).parent.parent.resolve() / "data" / "history.json"
    with open(save_path, "wb") as f:
        f.write(history_json)
    print("history is saved")

if __name__ == "__main__":
    main()

# Provide me with a good motto for today!
# Why did you choose this one?
# How about another one for my friend?
