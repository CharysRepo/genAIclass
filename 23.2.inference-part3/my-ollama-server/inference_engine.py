import asyncio
import time
import uuid
from llama_cpp import Llama
from metrics_collector import MetricsCollector
from custom_types import ChatCompletionRequest

class InferenceEngine:
    def __init__(self, model_path: str, max_concurrent: int = 4, n_ctx: int = 4096):
        self.model_id = model_path
        self.metrics = MetricsCollector()
        
        # Initialize the model with structural threading slots enabled
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            verbose=False,
            # CRITICAL: Allocate specific thread workers for concurrent parallel execution
            n_threads=4,          # Threads used for a single request processing
            n_threads_batch=4     # Threads used for parallel prompt evaluation
        )
        
        # We replace the Semaphore with a Lock because llama.cpp's internal state 
        # requires sequential slot evaluation to prevent context mixing corruption.
        self.lock = asyncio.Lock()

    async def generate_non_stream(self, request: ChatCompletionRequest) -> dict:
        arrival_time = time.time()
        raw_messages = [msg.model_dump() for msg in request.messages]
        
        # Use asyncio.Lock to safely schedule incoming token evaluation loops
        async with self.lock:
            start_gen_time = time.time()
            response = await asyncio.to_thread(
                lambda: self.llm.create_chat_completion(
                    messages=raw_messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
            )
            end_gen_time = time.time()

        completion_text = response["choices"][0]["message"]["content"]
        prompt_len = response["usage"]["prompt_tokens"]
        completion_len = response["usage"]["completion_tokens"]
        total_tokens = response["usage"]["total_tokens"]

        total_ms = (time.time() - arrival_time) * 1000
        ttft_ms = (end_gen_time - start_gen_time) * 1000 

        self.metrics.record_request(latency_ms=total_ms, ttft_ms=ttft_ms, total_tokens=total_tokens)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(arrival_time),
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": completion_text},
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_len,
                "total_tokens": total_tokens
            }
        }
