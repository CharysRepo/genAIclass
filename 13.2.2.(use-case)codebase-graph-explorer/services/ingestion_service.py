import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from services.neo4j_db_service import AsyncNeo4jDBService
from ast_parser.parser import parse_codebase_directory

class AsyncIngestionService:
    def __init__(self, db_service: AsyncNeo4jDBService):
        """Injects the async database engine dependency adapter."""
        self.db = db_service
        # Controls maximum concurrent database connections to avoid overwhelming the driver pool
        self.batch_size = 100 

    @staticmethod
    def _flatten_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
        """Converts lists and nested configuration shapes into flat Neo4j scalars."""
        flat_props = {}
        for k, v in properties.items():
            if isinstance(v, (list, tuple)):
                flat_props[k] = [str(item) for item in v]
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat_props[f"{k}_{sub_k}"] = str(sub_v) if isinstance(sub_v, (list, dict)) else sub_v
            else:
                flat_props[k] = v
        return flat_props

    def _determine_labels(self, node_type: str, flat_props: Dict[str, Any]) -> str:
        """Evaluates payload characteristics to format sanitized multi-label signatures."""
        labels = ["CodeNode", node_type]
        
        if flat_props.get("is_async") in ("True", True):
            labels.append("Async")
        if "Method" in node_type or  "Function" in node_type:
            labels.append("Callable")
        if "Module" in node_type:
            labels.append("File")
            
        return ":".join([f"`{lbl.strip()}`" for lbl in labels])

    async def _ingest_nodes_concurrently(self, nodes: List[Dict[str, Any]]):
        """Processes and streams node structures concurrently using chunked batch tasks."""
        tasks = []
        for node in nodes:
            node_id = node["node_id"]
            node_type = node.get("node_type", "Unknown")
            
            flat_props = self._flatten_properties(node.get("properties", {}))
            flat_props["id"] = node_id
            flat_props["name"] = node.get("name", "Unknown")
            flat_props["code"] = node.get("code_snippet", "")
            flat_props["line_no"] = node.get("line_no", 0)
            flat_props["end_lineno"] = node.get("end_lineno", 0)

            
            labels_string = self._determine_labels(node_type, flat_props)
            
            # Queue up the coroutine function without awaiting it immediately
            tasks.append(self.db.create_indexed_node(node_id, labels_string, flat_props))

            # Fire execution once the batch ceiling is reached
            if len(tasks) >= self.batch_size:
                await asyncio.gather(*tasks)
                tasks = []
                
        # Drain any remaining queued records
        if tasks:
            await asyncio.gather(*tasks)

    async def _ingest_relationships_concurrently(self, relationships: List[Dict[str, Any]]):
        """Ensures integrity anchors exist and connects edges concurrently in chunks."""
        # Step A: Aggregate all required node IDs to build safe database stubs first
        stub_ids = set()
        for rel in relationships:
            stub_ids.add(rel["source"])
            stub_ids.add(rel["target"])
            
        stub_tasks = [self.db.merge_stub_node(sid) for sid in stub_ids]
        
        # Batch execute the creation of missing skeletons 
        for i in range(0, len(stub_tasks), self.batch_size):
            await asyncio.gather(*stub_tasks[i:i + self.batch_size])

        # Step B: Link edges concurrently
        rel_tasks = []
        for rel in relationships:
            source = rel["source"]
            target = rel["target"]
            rel_type = rel["rel_type"].upper()
            flat_rel_props = self._flatten_properties(rel.get("properties", {}))
            
            rel_tasks.append(self.db.create_relationship(source, target, rel_type, flat_rel_props))
            
            if len(rel_tasks) >= self.batch_size:
                await asyncio.gather(*rel_tasks)
                rel_tasks = []
                
        if rel_tasks:
            await asyncio.gather(*rel_tasks)

    async def ingest_codebase(self, dir: str, clear_first: bool = True):
        """Asynchronously coordinates full file schema ingestion workflows sequentially."""
        if clear_first:
            print("🧹 Clearing database tracking layers asynchronously...")
            await self.db.clear_database()

        graph_data = parse_codebase_directory(dir)

        # Step 1: Execute concurrent node ingestion
        nodes = graph_data.get("nodes", [])
        print(f"📥 Concurrently processing {len(nodes)} infrastructure nodes...")
        await self._ingest_nodes_concurrently(nodes)
        print("✅ Multi-label structural nodes ingested successfully.")

        # Step 2: Execute concurrent relationship binding
        relationships = graph_data.get("relationships", [])
        print(f"🔗 Concurrently connecting {len(relationships)} cross-module network edges...")
        await self._ingest_relationships_concurrently(relationships)
        print("✅ Cross-module connection links mapped successfully.")

        print("🎉 Concurrent async graph architecture ingestion run complete!")
