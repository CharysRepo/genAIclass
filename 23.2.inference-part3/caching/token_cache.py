from pathlib import Path
import time
from llama_cpp import Llama

def illustrate_kv_cache_isolation(model_path: Path):
    print("📦 Loading Model Instance...")
    model = Llama(
        model_path=str(model_path),
        n_ctx=8192,
        n_gpu_layers=-1,
        verbose=False
    )
    
    # Use a fixed, medium-sized prompt to establish a baseline processing load
    prompt = "Explain the fundamental principles of quantum computing and how superposition works."
    
    print("\n🚀 Executing Generation Stream to Isolate KV Cache Behavior...")
    
    # We will generate a continuous stream of tokens, capturing timestamps for each token
    token_timestamps = []    
    tokens_generator = model.create_completion(
        prompt=prompt,
        max_tokens=250,
        temperature=0.3,
        stream=True
    )
    
    # Track the exact moment generation is initialized
    start_stream = time.perf_counter()
    
    for chunk in tokens_generator:
        print(chunk["choices"][0]["text"])
        token_arrival = time.perf_counter()
        token_timestamps.append(token_arrival)

    # Calculate token intervals
    # Token 1 interval includes the raw prompt prefill math (building the KV Cache)
    t1_time = token_timestamps[0] - start_stream
    
    # Subsequent tokens reuse the KV cache (incremental generation / decoding)
    subsequent_times = [
        token_timestamps[i] - token_timestamps[i-1] 
        for i in range(1, len(token_timestamps))
    ]
    avg_subsequent_time = sum(subsequent_times) / len(subsequent_times)
    
    # Calculate performance metrics
    prefill_speed = 1 / t1_time
    decoding_speed = 1 / avg_subsequent_time
    kv_cache_efficiency_multiplier = t1_time / avg_subsequent_time


    print("\n📊 ISOLATED KV CACHE ANALYSIS:")
    print(f"• Token #1 Time (Prefill / Building KV Cache):   {t1_time:.4f}s ({prefill_speed:.2f} tokens/sec)")
    print(f"• Tokens #2-{len(token_timestamps)} Avg Time (Decoding / Reusing KV Cache): {avg_subsequent_time:.4f}s ({decoding_speed:.2f} tokens/sec)")
    print(f"⚡ The KV Cache made incremental generation {kv_cache_efficiency_multiplier:.1f}x FASTER than initial processing!")
    
if __name__ == "__main__":
    GGUF_FILE = Path("F:/gguf/model_q4_k_m.gguf")
    if GGUF_FILE.exists():
        illustrate_kv_cache_isolation(GGUF_FILE)
    else:
        print(f"❌ Model file not found at: {GGUF_FILE}")
