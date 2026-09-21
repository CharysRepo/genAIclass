import os
os.environ["HF_HOME"] = "F:/hf"
from pathlib import Path
from torchinfo import summary
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch

def get_inference_pipeline(checkpoint):    
    model_dir = "HuggingFaceTB/SmolLM2-1.7B-Instruct"    
    base_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    peft_model = PeftModel.from_pretrained(base_model, checkpoint)
    print(peft_model)
    summary(peft_model)
    merged_model = peft_model.merge_and_unload()
    print(merged_model)
    summary(merged_model)

    return pipeline(
        "text-generation", 
        model=merged_model, 
        tokenizer=tokenizer,
        device_map="auto"
    )

def infer(pipe, system_input:str, user_input: str) -> str:
    chat_prompt = [
        {"role": "system", "content": system_input},
        {"role": "user", "content": user_input}
    ]

    outputs = pipe(
        chat_prompt, 
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.1
    )
    
    return outputs[0]['generated_text'][-1]["content"]

if __name__ == "__main__":
    adapter_dir = Path(__file__).parent.resolve() / "smollm2-1.7b-it-function-calling-adapter-lora" / "checkpoint-21"
    gen_pipe = get_inference_pipeline(adapter_dir)

    system_input = "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.\n<tools>\n[{\"type\": \"function\", \"function\": {\"name\": \"search_flights\", \"description\": \"Searches for flights based on departure and destination cities, dates, class, and other preferences.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"departure_city\": {\"type\": \"string\", \"description\": \"The city from which the flight will depart.\"}, \"destination_city\": {\"type\": \"string\", \"description\": \"The destination city for the flight.\"}, \"departure_date\": {\"type\": \"string\", \"description\": \"The departure date for the flight.\", \"format\": \"date\"}, \"return_date\": {\"type\": \"string\", \"description\": \"The return date for the flight.\", \"format\": \"date\"}, \"class\": {\"type\": \"string\", \"description\": \"The class of the flight ticket.\", \"enum\": [\"economy\", \"business\", \"first\"]}, \"flexible_cancellation\": {\"type\": \"boolean\", \"description\": \"Indicates if the search should filter for flights with flexible cancellation policies.\"}}, \"required\": [\"departure_city\", \"destination_city\", \"departure_date\", \"return_date\", \"class\"]}}}]\n</tools>\nFor each function call return a json object with function name and arguments within <tool_call> </tool_call> tags with the following schema:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-dict>}\n</tool_call>\n"
    user_input = "I'm planning a kayaking trip and looking to book flights from Los Angeles to Auckland. My departure is scheduled for July 10th, 2023, and I intend to return on July 24th, 2023. I would prefer to travel in economy class and would also like the option to have flexible cancellation policies for the tickets due to the uncertain nature of outdoor activities. Could you please search for flights that meet these criteria and provide me with the available options?"
    result = infer(gen_pipe, system_input, user_input)    
    print(f"Response:\n{result}")

