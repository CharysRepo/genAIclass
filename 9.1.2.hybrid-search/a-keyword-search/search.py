import asyncio
import asyncpg

async def run_search(pool, user_input, limit=3):    
    async with pool.acquire() as conn:
        parsed_query = await conn.fetchval("SELECT plainto_tsquery('english', $1)::text", user_input)
        print(f"PostgreSQL processed '{user_input}' into: {parsed_query}")

        query = """
            SELECT doc_name, ts_rank(search_vector, query) as score
            FROM documents1, plainto_tsquery('english', $1) query
            WHERE search_vector @@ query
            ORDER BY score DESC
            LIMIT $2;  
        """
        
        rows = await conn.fetch(query, user_input, limit)

        print(f"Results for '{user_input}':")
        if not rows:
            print("No matches found.")
        else:
            for row in rows:
                print(f"doc_name={row['doc_name']} | Rank={row['score']:.4f}")


async def main():
    DSN = "postgresql://postgres:postgres@localhost:5432/postgres"
    pool = await asyncpg.create_pool(DSN)
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