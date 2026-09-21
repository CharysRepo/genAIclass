import time
from pathlib import Path
from llama_cpp import Llama

def run_system_prompt_only_cache_benchmark(model):
    # A heavy, token-dense system prompt to make the caching speedup clearly visible
    SYSTEM_PROMPT = (
        "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. "
        "Here are the available tools:<tools> "
        "[{'type': 'function', 'function': {'name': 'get_stock_price', 'description': 'Get current stock price', 'parameters': {'type': 'object', 'properties': {'company': {'type': 'string'}}, 'required': ['company']}}}, "
        "{'type': 'function', 'function': {'name': 'get_movie_details', 'description': 'Get movie details', 'parameters': {'type': 'object', 'properties': {'title': {'type': 'string'}}, 'required': ['title']}}}] "
        "</tools> Return json within <tool_call></tool_call> tags."
    )
    
    # Distinct, isolated user queries (completely fresh sessions each time)
    independent_queries = [
        "What is the current stock price of Apple right now?",       # Turn 1: Cold start (Processes System Prompt + Query 1)
        "Can you look up the cast and plot details for Inception?",   # Turn 2: Cache Hit (Reuses System Prompt)
        "How much is a single share of Microsoft trading for?",       # Turn 3: Cache Hit (Reuses System Prompt)
        "Give me the director and runtime for the movie Interstellar." # Turn 4: Cache Hit (Reuses System Prompt)
    ]
    
    execution_times = []
    
    print("\n🚀 Starting System Prompt Cache Isolation Benchmark...")
    print(f"📋 System Prompt Word Count: ~{len(SYSTEM_PROMPT.split())} words")

    for i, user_query in enumerate(independent_queries):
        clean_chat_payload = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
        
        print(f"\n▶️ [SESSION {i+1}] User Query: '{user_query}'")
        
        start_time = time.perf_counter()
        response = model.create_chat_completion(
            messages=clean_chat_payload,
            max_tokens=50,
            temperature=0.0
        )
        elapsed = time.perf_counter() - start_time
        execution_times.append(elapsed)
        
        assistant_text = response["choices"][0]["message"]["content"].strip()
        print(f"🤖 Assistant: {assistant_text}")

    return execution_times

if __name__ == "__main__":
    # Update with your local GGUF path
    gguf_file = Path("F:/gguf/model_q4_k_m.gguf")
    
    if not gguf_file.exists():
        print(f"❌ Error: Model file not found at {gguf_file}.")
        exit(1)
        
    print("📦 Building Model Instance ...")
    model = Llama(model_path=str(gguf_file), n_ctx=8192, n_gpu_layers=-1, verbose=False)
    
    # Execute the test and capture durations
    times = run_system_prompt_only_cache_benchmark(model)
    
    print("\n📊 SYSTEM PROMPT CACHE PERFORMANCE ANALYSIS:")
    print(f" • Session 1 (Cold Start):   {times[0]:.4f}s")
    print(f" • Session 2 (Cache Hit 1):  {times[1]:.4f}s")
    print(f" • Session 3 (Cache Hit 2):  {times[2]:.4f}s")
    print(f" • Session 4 (Cache Hit 3):  {times[3]:.4f}s")
    print("-" * 50)
    
    # Calculate performance improvements against the baseline cold start
    cold_start_time = times[0]
    for idx, cache_hit_time in enumerate(times[1:], start=2):
        speedup = ((cold_start_time - cache_hit_time) / cold_start_time) * 100
        print(f" ⚡ Session {idx} was {speedup:.1f}% faster than the Cold Start baseline.")

    print("\n🏁 Isolated system prompt benchmark complete.")
