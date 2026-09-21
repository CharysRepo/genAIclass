import os
os.environ["HF_HOME"] = "F:/hf"
from pathlib import Path
from llama_cpp import Llama

def load_gguf_model(gguf_path: Path):    
    model = Llama(
        model_path=str(gguf_path),
        n_ctx=8096,
        n_gpu_layers=-1,
        verbose=False 
    )
    return model

def infer_gguf(model, system_input: str, user_input: str) -> str:
    chat_prompt = [
        {"role": "system", "content": system_input},
        {"role": "user", "content": user_input}
    ]
    response = model.create_chat_completion(
        messages=chat_prompt,
        max_tokens=256,
        temperature=0.1,
        repeat_penalty=1.1
    )
    
    # Extract the string content out of the structured OpenAI-style output
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    gguf_file = "F:/gguf/model_q4_k_m.gguf"
    gguf_model = load_gguf_model(gguf_file)

    system_input = "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.Here are the available tools:<tools> [{'type': 'function', 'function': {'name': 'get_stock_price', 'description': 'Get the current stock price of a company', 'parameters': {'type': 'object', 'properties': {'company': {'type': 'string', 'description': 'The name of the company'}}, 'required': ['company']}}}, {'type': 'function', 'function': {'name': 'get_movie_details', 'description': 'Get details about a movie', 'parameters': {'type': 'object', 'properties': {'title': {'type': 'string', 'description': 'The title of the movie'}}, 'required': ['title']}}}] </tools>Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{tool_call}\n</tool_call>"
    user_input = "can you tell me the current stock price of Apple?"
    result = infer_gguf(gguf_model, system_input, user_input)    
    print(f"Response:\n{result}")

