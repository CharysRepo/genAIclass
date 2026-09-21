from pydantic_ai import Embedder
import asyncio
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
    id: int
    sent1: str
    sent2: str
    score: float

# 2. Function to calculate cosine similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

async def main():
    data = [
        ("A girl is styling her hair.", "A girl is brushing her hair."),
        ("A group of men play soccer on the beach.", "A group of boys are playing soccer on the beach."),
        ("One woman is measuring another woman's ankle.", "A woman measures another woman's ankle."),
        ("A man is cutting up a cucumber.", "A man is slicing a cucumber."),
        ("A man is playing a harp.", "A man is playing a keyboard."),
        ("A woman is cutting onions.", "A woman is cutting tofu."),
        ("A man is riding an electric bicycle.", "A man is riding a bicycle."),
        ("A man is playing the drums.", "A man is playing the guitar."),
        ("A man is playing guitar.", "A lady is playing the guitar."),
        ("A man is playing a guitar.", "A man is playing a trumpet.")
    ]

    print(f"{'sent1':<50} {'sent2':<50} {'score':<6}")

    results = []
    for i, (s1, s2) in enumerate(data):
        result = await embedder.embed_query([s1, s2])
        v1, v2 = result.embeddings[0], result.embeddings[1]

        score = cosine_similarity(v1, v2)
        
        output = SimilarityResult(id=i, sent1=s1, sent2=s2, score=score)
        results.append(f"{output.id} {output.sent1:<50} {output.sent2:<50} {output.score:.4f}")
    print("\n".join(results))

if __name__ == "__main__":
    asyncio.run(main())