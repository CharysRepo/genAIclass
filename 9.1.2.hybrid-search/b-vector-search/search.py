import asyncio
import asyncpg
from pgvector.asyncpg import register_vector
from pydantic_ai import Embedder
from dotenv import load_dotenv

load_dotenv(override=True)
embedder = Embedder('ollama:snowflake-arctic-embed2')

async def run_search(pool, user_input, limit=3):    
    async with pool.acquire() as conn:
        result = await embedder.embed_query(user_input)
        query_vector = result.embeddings[0]
        print(query_vector)

        query = """
            SELECT doc_name, doc_content, (embedding_vector <=> $1) AS score
            FROM documents2
            ORDER BY score
            LIMIT $2; 
        """
        rows = await conn.fetch(query, query_vector, limit)

        print(f"Results for '{user_input}':")
        if not rows:
            print("No matches found.")
        else:
            for row in rows:
                print(f"doc_name={row['doc_name']} | Rank={row['score']:.4f}")

async def main():
    DSN = "postgresql://postgres:postgres@localhost:5432/postgres"
    pool = await asyncpg.create_pool(DSN, init=register_vector)
    while True:
        user_input = input("User:")
        if user_input == "quit":
            break
        await run_search(pool, user_input)
    await pool.close()

if __name__ == '__main__':
    asyncio.run(main())

# Can I use ChatGPT for my assignment?
# What happens if I get sick during an exam?
# How is the AI & Machine Learning course graded?