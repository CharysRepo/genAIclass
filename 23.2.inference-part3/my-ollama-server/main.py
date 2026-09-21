import os
os.environ["HF_HOME"] = "F:/hf"
import uvicorn
from fastapi import FastAPI
from custom_types import ChatCompletionRequest
from inference_engine import InferenceEngine

app = FastAPI(title="My ollama server")
engine = InferenceEngine("F:/gguf/model_q4_k_m.gguf", max_concurrent=4)

@app.post("/chat/completions")
async def chat_completions(payload: ChatCompletionRequest):
    return await engine.generate_non_stream(payload)
    
@app.get("/metrics")
def get_metrics():
    return engine.metrics.get_current_metrics()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)

