import asyncio
import asyncpg
from pgvector.asyncpg import register_vector
from pydantic_ai import Embedder
from dotenv import load_dotenv

load_dotenv(override=True)
embedder = Embedder('ollama:snowflake-arctic-embed2')

async def run_search(pool, user_input, limit=10, k=60):    
    async with pool.acquire() as conn:
        result = await embedder.embed_query(user_input)
        query_vector = result.embeddings[0]

        vector_search_query = """
            SELECT doc_name, doc_content, (embedding_vector <=> $1) AS score
            FROM documents3
            ORDER BY score
            LIMIT $2; 
        """
        vector_rows = await conn.fetch(vector_search_query, query_vector, limit)
        print(vector_rows)

        kw_search_query = """
            SELECT doc_name, ts_rank(search_vector, query) as score
            FROM documents3, plainto_tsquery('english', $1) query
            WHERE search_vector @@ query
            ORDER BY score DESC
            LIMIT $2;  
        """
        kw_rows = await conn.fetch(kw_search_query, user_input, limit)
        print(kw_rows)

        scores = {}
        for rank, row in enumerate(vector_rows, start=1):
            name = row['doc_name']
            scores[name] = scores.get(name, 0) + (1 / (k + rank))

        for rank, row in enumerate(kw_rows, start=1):
            name = row['doc_name']
            scores[name] = scores.get(name, 0) + (1 / (k + rank))

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

        print(f"Results for '{user_input}':")
        if not sorted_results:
            print("No matches found.")
        else:
            for doc_name, score in sorted_results:
                print(f"doc_name={doc_name} | RRF Score={score:.4f}")

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