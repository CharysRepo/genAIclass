import asyncio
import time
import asyncpg

DB_CONFIG = {
    "database": "test",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

async def setup_social_network():
    async with asyncpg.create_pool(**DB_CONFIG, min_size=2, max_size=10) as pool:
        async with pool.acquire() as conn:
            async with conn.transaction():
                
                print("Creating 'users' table if it does not exist...")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(50),
                        age INT
                    );
                """)
                
                print("Creating 'friendships' table if it does not exist...")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS friendships (
                        user_id_1 INT REFERENCES users(id),
                        user_id_2 INT REFERENCES users(id),
                        PRIMARY KEY (user_id_1, user_id_2)
                    );
                """)
                
                print("Creating index on user_id_2 if it does not exist...")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_friendships_user2 ON friendships(user_id_2);")
                
                print("Inserting user records (ignoring duplicates)...")
                users_data = [
                    (1, 'Alice', 28),
                    (2, 'Bob', 31),
                    (3, 'Charlie', 25),
                    (4, 'David', 34),
                    (5, 'Emma', 29)
                ]
                await conn.executemany("""
                    INSERT INTO users (id, name, age) 
                    VALUES ($1, $2, $3)
                """, users_data)
                
                print("Inserting friendship records (ignoring duplicates)...")
                friendships_data = [
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5)
                ]
                await conn.executemany("""
                    INSERT INTO friendships (user_id_1, user_id_2) 
                    VALUES ($1, $2)
                    ON CONFLICT (user_id_1, user_id_2) DO NOTHING;
                """, friendships_data)
                
                print("Database setup complete!")

async def run_multi_hop_query():
    target_user_id = 1
    
    multi_hop_sql = """
        SELECT DISTINCT u4.name 
        FROM friendships f1
        JOIN friendships f2 ON f1.user_id_2 = f2.user_id_1
        JOIN friendships f3 ON f2.user_id_2 = f3.user_id_1
        JOIN friendships f4 ON f3.user_id_2 = f4.user_id_1
        JOIN users u4 ON f4.user_id_2 = u4.id
        WHERE f1.user_id_1 = $1;
    """

    print(f"Connecting to database to find 4-degree recommendations for User ID: {target_user_id}...")
    
    async with asyncpg.create_pool(**DB_CONFIG, min_size=1, max_size=5) as pool:
        async with pool.acquire() as conn:            
            start_time = time.perf_counter()            
            rows = await conn.fetch(multi_hop_sql, target_user_id)            
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            
            print("\n--- Query Results ---")
            if not rows:
                print("No 4-degree friends found.")
            else:
                for row in rows:
                    print(f"Recommended Friend: {row['name']}")
            print("---------------------")
            print(f"Query Execution Time: {execution_time_ms:.3f} ms\n")

if __name__ == "__main__":
    asyncio.run(setup_social_network())
    asyncio.run(run_multi_hop_query())
