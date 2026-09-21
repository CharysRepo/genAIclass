from pydantic import BaseModel
from pydantic_ai import Agent
import logfire
from dotenv import load_dotenv
from typing import Annotated

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

class UserProfile(BaseModel):
    name: Annotated[str, "The user's full name"]
    age: Annotated[int, "The user's age in years"]
    favorite_color: Annotated[str, "The user's preferred color name"]

agent = Agent(
    'groq:llama-3.3-70b-versatile',
    output_type=UserProfile | str,
    instructions=(
        "Extract user profile information."
        "If any fields are missing, do not guess. Instead, ask the user to provide the missing details."
    )
)

def collect_info():
    history = []
    user_input = "My name is John." # Initial partial data
    
    while True:
        result = agent.run_sync(user_input, message_history=history)
        
        if isinstance(result.output, UserProfile):
            print(f"Final Data Collected: {result.output}")
            break
        else:
            print(f"Agent: {result.output}")
            user_input = input("You: ")
            history.extend(result.new_messages())

collect_info()