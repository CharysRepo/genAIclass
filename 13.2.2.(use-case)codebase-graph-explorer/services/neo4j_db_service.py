import asyncio
import re
from neo4j import AsyncGraphDatabase
from typing import Dict, Any
from config.config_reader import settings

class AsyncNeo4jDBService:
    def __init__(self, uri: str=settings.NEO4J_URI, user: str=settings.NEO4J_USER, password: str=settings.NEO4J_PASSWORD, database: str=settings.NEO4J_DATABASE):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.database = database  

    async def close(self):
        if self.driver:
            await self.driver.close()

    async def clear_database(self):
        async with self.driver.session(database=self.database) as session:
            await session.run("MATCH (n) DETACH DELETE n")

    async def create_database_if_not_exists(self, target_database: str):
        if target_database.lower() == "system":
            return            
        try:
            async with self.driver.session(database="system") as session:
                query = f"CREATE DATABASE {target_database} IF NOT EXISTS"
                await session.run(query)
                print(f"✨ Checked/Created Neo4j database: '{target_database}'")
        except Exception as e:
            print(f"❌ Failed administrative database setup for '{target_database}': {e}")
            raise

    async def wait_for_database_online(self, target_database: str, timeout_seconds: int = 10):
        """Polls the system database until the target database state equals 'online'."""
        if target_database.lower() == "system":
            return

        start_time = asyncio.get_event_loop().time()
        query = "SHOW DATABASE $db_name YIELD currentStatus WHERE currentStatus = 'online'"
        
        print(f"⏳ Waiting for database '{target_database}' to come online...")
        
        while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
            try:
                async with self.driver.session(database="system") as session:
                    result = await session.run(query, db_name=target_database)
                    record = await result.single()
                    if record:
                        print(f"🟢 Database '{target_database}' is now active and online.")
                        return
            except Exception:
                pass  
                
            await asyncio.sleep(0.5)
            
        raise TimeoutError(f"❌ Database '{target_database}' failed to transition online within {timeout_seconds}s.")


    async def create_indexes_and_constraints(self, labels: list[str] = None):
        if labels is None:
            labels = ["CodeNode", "Module", "Class", "Function", "AsyncFunction", "Method", "AsyncMethod", "LocalVariable", "ClassVariable", "GlobalVariable"]

        async with self.driver.session(database=self.database) as session:
            # 1. Create a Uniqueness Constraint on the base 'CodeNode' identifier
            # This prevents duplicates across the entire directory merge run
            print("⚡ Creating uniqueness constraint on :CodeNode(node_id)...")
            constraint_query = """
            CREATE CONSTRAINT CODENODE_ID_UNIQUE IF NOT EXISTS
            FOR (n:CodeNode)
            REQUIRE n.node_id IS UNIQUE
            """
            await session.run(constraint_query)

            # 2. Create Range Indexes for all specific structural sub-labels
            # This speeds up type-specific queries like MATCH (c:Class {id: ...})
            for label in labels:
                # Skip CodeNode since it already got a high-priority uniqueness index above
                if label == "CodeNode":
                    continue
                
                print(f"⚡ Creating range index on :{label}(name)...")
                index_query = f"""
                CREATE INDEX `{label}_name_idx` IF NOT EXISTS
                FOR (n:`{label}`)
                ON (n.name)
                """
                await session.run(index_query)
                
            print("✅ All database indexes and structural constraints are configured.")

    async def setup_database(self):
        await self.create_database_if_not_exists(self.database)
        await self.wait_for_database_online(self.database)
        await self.create_indexes_and_constraints()            

    async def _fetch_nodes_and_properties(self) -> list:
        """Queries Neo4j schema for node labels and their properties."""
        query = """
        MATCH (n)
        UNWIND labels(n) AS label
        UNWIND keys(n) AS key
        RETURN label, collect(DISTINCT key) AS properties
        """
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return await result.data()

    async def create_indexed_node(self, node_id: str, labels_string: str, properties: Dict[str, Any]):
        """Merges a node with dynamic multi-labels using non-blocking session execution."""
        query = f"""
        MERGE (n:{labels_string} {{id: $node_id}})
        SET n += $properties
        """
        async with self.driver.session(database=self.database) as session:
            await session.run(query, node_id=node_id, properties=properties)

    async def merge_stub_node(self, node_id: str):
        """Asynchronously registers tracking skeletons for out-of-scope elements."""
        query = "MERGE (n {id: $node_id}) ON CREATE SET n.name = $node_id"
        async with self.driver.session(database=self.database) as session:
            await session.run(query, node_id=node_id)

    async def create_relationship(self, source_id: str, target_id: str, rel_type: str, properties: Dict[str, Any]):
        """Binds two nodes together asynchronously using a specialized directional edge."""
        query = f"""
        MATCH (s {{id: $source_id}}), (t {{id: $target_id}})
        MERGE (s)-[r:`{rel_type}`]->(t)
        SET r += $properties
        """
        async with self.driver.session(database=self.database) as session:
            await session.run(query, source_id=source_id, target_id=target_id, properties=properties)

    def _prepare_query(self, query: str) -> str:
        """Strips comments and normalizes whitespace."""
        match = re.search(r"```(?:cypher|neo4j)?(.*?)```", query, re.DOTALL)
        if match:
            clean_query = match.group(1).strip()
            return clean_query
        return query

    async def execute_cypher(self, query: str):
        try:
            clean_query = self._prepare_query(query)
            async with self.driver.session(database=self.database) as session:
                result = await session.run(clean_query)
                records = await result.data()
                return records
        except Exception as e:
            print(f"Error while executing cypher query: {e}")
            raise
