import asyncio
import time
import httpx

# Configuration
SERVER_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{SERVER_URL}/chat/completions"
METRICS_ENDPOINT = f"{SERVER_URL}/metrics"

async def send_chat_request(client: httpx.AsyncClient, request_id: int):
    """Sends a single chat completion request to the server."""
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Hello! Give me a short 2-sentence poem about coding. Tag: {request_id}"}
        ],
        "max_tokens": 50,
        "temperature": 0.7,
        "repetition_penalty": 1.1
    }
    
    start_time = time.time()
    try:
        response = await client.post(CHAT_ENDPOINT, json=payload, timeout=60.0)
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            reply = data["choices"][0]["message"]["content"].strip()
            tokens = data["usage"]["total_tokens"]
            print(f"[Req {request_id} Success] {latency:.0f}ms | Tokens: {tokens} | Reply: {reply}")
        else:
            print(f"[Req {request_id} Error] Status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"[Req {request_id} Failed] Exception: {e}")

async def fetch_metrics(client: httpx.AsyncClient):
    """Fetches and displays the performance metrics from the server."""
    try:
        response = await client.get(METRICS_ENDPOINT)
        if response.status_code == 200:
            print("\n=== Server Metrics ===")
            print(response.json())
        else:
            print(f"\nCould not fetch metrics. Status: {response.status_code}")
    except Exception as e:
        print(f"\nFailed to fetch metrics: {e}")

async def main():
    # Use a connection pool to handle multiple concurrent requests cleanly
    async with httpx.AsyncClient() as client:
        n_concurrent_req = 5
        print(f"Sending {n_concurrent_req} concurrent requests to test ...")
        
        # Trigger 5 requests at the exact same time
        tasks = [send_chat_request(client, i) for i in range(1, n_concurrent_req+1)]
        await asyncio.gather(*tasks)
        
        # Fetch the collector stats to verify performance
        await fetch_metrics(client)

if __name__ == "__main__":
    asyncio.run(main())
