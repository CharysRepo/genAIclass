from pydantic_ai import Embedder
import asyncio
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

async def main():
    queries = ['Hello world', "how are you?"]
    result = await embedder.embed_query(queries)
    print(result.embeddings[0])
    print(result.embeddings[1])
    #print(result['Hello world'])
    #print(result["how are you?"])
    print(f"Embedding dimension: {len(result.embeddings[0])}")
    print(f"Tokens used: {result.usage}")

    docs = [
        'Machine learning is a subset of AI.',
        'Deep learning uses neural networks.',
        'Python is a programming language.',
    ]
    result = await embedder.embed_documents(docs)
    #print(result.embeddings)
    print(f'Embedded {len(result.embeddings)} documents')



if __name__ == "__main__":
    asyncio.run(main())
