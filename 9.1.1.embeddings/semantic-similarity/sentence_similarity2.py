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
    sent1: str
    sent2: str
    score: float

# 2. Function to calculate cosine similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

async def main():
    messages = [
            "A man is eating food.",
            "A man is eating a piece of bread.",
            "The girl is carrying a baby.",
            "A man is riding a horse.",
            "A woman is playing violin.",
            "Two men pushed carts through the woods.",
            "A man is riding a white horse on an enclosed ground.",
            "A monkey is playing drums.",
            "A cheetah is running behind its prey.",
            "technology is changing the facet of world",
            "technology also have side effects",
            "today is a sunny day",
            "i love coding",
            "coding is my passion",
            "the cat sat on the mat",
            "dogs are awesome",
            "the weather today is beautiful",
            "it is raining",
            "the weather is lovely today",
    ]
    results = await embedder.embed_query(messages)
    embeddings = results.embeddings 

    print(f"{'Index':<5} {'Sent 1':<45} {'Sent 2':<45} {'Score'}")

    results = []
    for i, j in combinations(range(len(messages)), 2):
        s1, s2 = messages[i], messages[j]
        v1, v2 = embeddings[i], embeddings[j]
        
        score = cosine_similarity(v1, v2)
        
        # Validate with Pydantic model
        result = SimilarityResult(sent1=s1, sent2=s2, score=score)
        
        # Display output
        results.append(f"{i}-{j:<3} {result.sent1:<45} {result.sent2:<45} {result.score:.4f}")

    print("\n".join(results))

if __name__ == "__main__":
    asyncio.run(main())