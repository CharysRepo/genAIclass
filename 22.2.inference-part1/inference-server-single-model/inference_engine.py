import asyncio
import time
import uuid
from metrics_collector import MetricsCollector
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline
from custom_types import ChatCompletionRequest

class InferenceEngine:
    def __init__(self, model_id: str, max_concurrent: int = 4):
        self.model_id = model_id
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.metrics = MetricsCollector()        
        self.pipe = self._get_inference_pipeline(model_id)

    def _get_inference_pipeline(self, model_path: str):    
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            device_map="auto"
        )
        

    async def generate_non_stream(self, request: ChatCompletionRequest) -> dict:
        arrival_time = time.time()

        raw_messages = [msg.model_dump() for msg in request.messages]
        async with self.semaphore:
            outputs = await asyncio.to_thread(
                lambda: self.pipe(
                    raw_messages, 
                    max_new_tokens=request.max_tokens,
                    do_sample=request.do_sample,
                    temperature=request.temperature,
                    repetition_penalty=request.repetition_penalty
                )
            )
        completion_text = outputs[0]['generated_text'][-1]["content"]

        # compute metrics
        total_ms = (time.time() - arrival_time) * 1000
        res = self.pipe.tokenizer.apply_chat_template(raw_messages, tokenize=True, add_generation_prompt=False)
        prompt_len = len(res.input_ids)
        completion_tokens = self.pipe.tokenizer.encode(completion_text, add_special_tokens=False)
        completion_len = len(completion_tokens)
        self.metrics.record_request(latency_ms=total_ms, ttft_ms=total_ms, total_tokens=prompt_len + completion_len)

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
                "total_tokens": prompt_len + completion_len
            }
        }