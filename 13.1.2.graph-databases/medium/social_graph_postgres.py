import asyncio
import random
import time
import asyncpg
from faker import Faker

DB_CONFIG = {
    "database": "test",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

NUM_USERS = 100000
NUM_FRIENDSHIPS = 300000

async def setup_large_postgres():
    fake = Faker()
    print(f"Generating data for {NUM_USERS} users...")
    
    # Generate mock users
    users_data = [(i, fake.first_name(), random.randint(18, 70)) for i in range(1, NUM_USERS + 1)]
    
    # Generate mock unique friendships
    friendships_set = set()
    while len(friendships_set) < NUM_FRIENDSHIPS:
        u1 = random.randint(1, NUM_USERS)
        u2 = random.randint(1, NUM_USERS)
        if u1 != u2:
            # Ensure order so (1,2) and (2,1) don't violate unique primary key constraints
            friendships_set.add((min(u1, u2), max(u1, u2)))
    friendships_data = list(friendships_set)

    async with asyncpg.create_pool(**DB_CONFIG, min_size=2, max_size=10) as pool:
        async with pool.acquire() as conn:
            async with conn.transaction():
                print("Resetting tables...")
                await conn.execute("DROP TABLE IF EXISTS friendships CASCADE;")
                await conn.execute("DROP TABLE IF EXISTS users CASCADE;")
                
                await conn.execute("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(50), age INT);")
                await conn.execute("""
                    CREATE TABLE friendships (
                        user_id_1 INT REFERENCES users(id),
                        user_id_2 INT REFERENCES users(id),
                        PRIMARY KEY (user_id_1, user_id_2)
                    );
                """)
                
                print("Inserting users in bulk...")
                await conn.executemany("INSERT INTO users (id, name, age) VALUES ($1, $2, $3);", users_data)
                
                print("Inserting friendships in bulk...")
                await conn.executemany("INSERT INTO friendships (user_id_1, user_id_2) VALUES ($1, $2);", friendships_data)
                
                print("Creating lookup index...")
                await conn.execute("CREATE INDEX idx_friendships_user2 ON friendships(user_id_2);")
                print("PostgreSQL setup complete!")

async def run_postgres_query():
    # Query checking combinations of directions since friendships are generated randomly
    multi_hop_sql = """
        SELECT DISTINCT u4.name 
        FROM friendships f1
        JOIN friendships f2 ON f1.user_id_2 = f2.user_id_1
        JOIN friendships f3 ON f2.user_id_2 = f3.user_id_1
        JOIN friendships f4 ON f3.user_id_2 = f4.user_id_1
        JOIN friendships f5 ON f4.user_id_2 = f5.user_id_1
        JOIN friendships f6 ON f5.user_id_2 = f6.user_id_1
        JOIN users u4 ON f4.user_id_2 = u4.id
        WHERE f1.user_id_1 = $1 LIMIT 100;
    """
    
    async with asyncpg.create_pool(**DB_CONFIG, min_size=1, max_size=5) as pool:
        async with pool.acquire() as conn:            
            start_time = time.perf_counter()            
            rows = await conn.fetch(multi_hop_sql, 1)            
            end_time = time.perf_counter()
            print(f"PostgreSQL 4-Hop Time: {(end_time - start_time) * 1000:.3f} ms (Found {len(rows)} samples)")

if __name__ == "__main__":
    asyncio.run(setup_large_postgres())
    asyncio.run(run_postgres_query())
