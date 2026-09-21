from pydantic_ai import Embedder
import asyncio
from itertools import combinations
from pydantic import BaseModel
import numpy as np
from dotenv import load_dotenv
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()
Embedder.instrument_all()

embedder = Embedder(
    #'github:openai/text-embedding-3-small'
    'ollama:snowflake-arctic-embed2'
    )

class SimilarityResult(BaseModel):
    word1: str
    word2: str
    score: float

# 2. Function to calculate cosine similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

async def main():
    words = [
        "rice",
        "attraction",
        "love",
        "friendship",
        "twitter",
        "queen",
        "king",
        "minister",
        "google",
        "search",
        "eating",
        "habbit",
        "food",
        "cook",
        "peace",
        "submarine",
        "war",
        "missile",
    ]

    results = await embedder.embed_query(words)
    embeddings = results.embeddings 

    print(f"{'Index':<5} {'Word 1':<20} {'Word 2':<20} {'Score'}")

    results = []
    for i, j in combinations(range(len(words)), 2):
        w1, w2 = words[i], words[j]
        v1, v2 = embeddings[i], embeddings[j]
        
        score = cosine_similarity(v1, v2)
        result = SimilarityResult(word1=w1, word2=w2, score=score)
        
        results.append(f"{i}-{j:<3} {result.word1:<20} {result.word2:<20} {result.score:.4f}")

    print("\n".join(results))

if __name__ == "__main__":
    asyncio.run(main())