import asyncio
import time
from neo4j import AsyncGraphDatabase

# Database connection configuration
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "auth": ("neo4j", "neo4j123")
}

async def setup_social_network():
    async with AsyncGraphDatabase.driver(**NEO4J_CONFIG) as driver:
        async with driver.session(database="test") as session:
            print("Creating uniqueness constraint on User(id) if it does not exist...")
            await session.run("""
                CREATE CONSTRAINT user_id_unique IF NOT EXISTS
                FOR (u:User) REQUIRE u.id IS UNIQUE;
            """)
            
            # Constraints automatically build look-up indexes. No separate index statement is needed.
            print("Inserting user records...")
            users_data = [
                {"id": 1, "name": "Alice", "age": 28},
                {"id": 2, "name": "Bob", "age": 31},
                {"id": 3, "name": "Charlie", "age": 25},
                {"id": 4, "name": "David", "age": 34},
                {"id": 5, "name": "Emma", "age": 29}
            ]
            
            await session.run("""
                UNWIND $users AS user
                MERGE (u:User {id: user.id})
                SET u.name = user.name, u.age = user.age;
            """, users=users_data)
            
            print("Inserting friendship records...")
            friendships_data = [
                {"source": 1, "target": 2},
                {"source": 2, "target": 3},
                {"source": 3, "target": 4},
                {"source": 4, "target": 5}
            ]
            
            # Batch matching and linking nodes
            await session.run("""
                UNWIND $friendships AS edge
                MATCH (source:User {id: edge.source})
                MATCH (target:User {id: edge.target})
                MERGE (source)-[:FRIEND]->(target);
            """, friendships=friendships_data)
            
            print("Database setup complete!")

async def run_multi_hop_query():
    target_user_id = 1
    
    # 4-Degree Friend Recommendation Cypher Query
    multi_hop_cypher = """
        MATCH (a:User {id: $target_id})-[:FRIEND*4]->(fof)
        RETURN DISTINCT fof.name AS name;
    """

    print(f"Connecting to database to find 4-degree recommendations for User ID: {target_user_id}...")
    
    async with AsyncGraphDatabase.driver(**NEO4J_CONFIG) as driver:
        async with driver.session(database="test") as session:            
            start_time = time.perf_counter()
            
            result = await session.run(multi_hop_cypher, target_id=target_user_id)
            end_time = time.perf_counter()
            records = await result.data()
                        
            execution_time_ms = (end_time - start_time) * 1000
            
            print("\n--- Query Results ---")
            if not records:
                print("No 4-degree friends found.")
            else:
                for record in records:
                    print(f"Recommended Friend: {record['name']}")
            print("---------------------")
            print(f"Query Execution Time: {execution_time_ms:.3f} ms\n")

if __name__ == "__main__":
    asyncio.run(setup_social_network())
    asyncio.run(run_multi_hop_query())
