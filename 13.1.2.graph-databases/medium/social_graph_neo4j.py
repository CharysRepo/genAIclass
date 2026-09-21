import asyncio
import random
import time
from neo4j import AsyncGraphDatabase
from faker import Faker

NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "auth": ("neo4j", "neo4j123")
}

NUM_USERS = 100000
NUM_FRIENDSHIPS = 300000

async def setup_large_neo4j():
    fake = Faker()
    print(f"Generating data for {NUM_USERS} users...")
    
    users_data = [{"id": i, "name": fake.first_name(), "age": random.randint(18, 70)} for i in range(1, NUM_USERS + 1)]
    
    friendships_set = set()
    while len(friendships_set) < NUM_FRIENDSHIPS:
        u1 = random.randint(1, NUM_USERS)
        u2 = random.randint(1, NUM_USERS)
        if u1 != u2:
            friendships_set.add((u1, u2))
    friendships_data = [{"source": f[0], "target": f[1]} for f in friendships_set]

    async with AsyncGraphDatabase.driver(**NEO4J_CONFIG) as driver:
        async with driver.session(database="test") as session:
            
            print("Cleaning existing graph data...")
            await session.run("MATCH (n) DETACH DELETE n;")
            
            print("Creating uniqueness constraint...")
            await session.run("CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;")
            
            # Neo4j handles large data best when broken into chunks during ingestion
            print("Streaming users into graph database in batches...")
            batch_size = 10000
            for i in range(0, len(users_data), batch_size):
                batch = users_data[i:i + batch_size]
                await session.run("""
                    UNWIND $batch AS user
                    CREATE (u:User {id: user.id, name: user.name, age: user.age});
                """, batch=batch)
                
            print("Streaming friendships into graph database in batches...")
            for i in range(0, len(friendships_data), batch_size):
                batch = friendships_data[i:i + batch_size]
                await session.run("""
                    UNWIND $batch AS edge
                    MATCH (source:User {id: edge.source})
                    MATCH (target:User {id: edge.target})
                    CREATE (source)-[:FRIEND]->(target);
                """, batch=batch)
                
            print("Neo4j setup complete!")

async def run_neo4j_query():
    # Traverses natively via physical pointers out to 4 degrees
    multi_hop_cypher = """
        MATCH (a:User {id: $target_id})-[:FRIEND*6]->(fof)
        RETURN DISTINCT fof.name AS name LIMIT 100;
    """
    
    async with AsyncGraphDatabase.driver(**NEO4J_CONFIG) as driver:
        async with driver.session(database="test") as session:
            start_time = time.perf_counter()
            result = await session.run(multi_hop_cypher, target_id=1)
            records = await result.data()
            end_time = time.perf_counter()
            print(f"Neo4j 4-Hop Time: {(end_time - start_time) * 1000:.3f} ms (Found {len(records)} samples)")

if __name__ == "__main__":
    asyncio.run(setup_large_neo4j())
    asyncio.run(run_neo4j_query())
