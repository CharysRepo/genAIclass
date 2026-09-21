import json
from pathlib import Path
from typing import Any, Dict
from ast_parser.code_graph_extractor import CodeGraphExtractor
from config.config_reader import settings

def build_package_map(root_path: Path) -> Dict[str, str]:
    """
    Scans the directory to map absolute file systems back to clean Python 
    import dot paths (e.g., 'src/utils/math.py' -> 'src.utils.math').
    """
    package_map = {}
    for file_path in root_path.rglob("*.py"):
        # Ignore virtual environments, caches, and hidden paths
        if any(part.startswith('.') or part in ('venv', '__pycache__', 'env', 'site-packages') for part in file_path.parts):
            continue
            
        # Create dot-notation module name relative to execution root
        relative_path = file_path.relative_to(root_path)
        if relative_path.name == "__init__.py":
            module_dot_path = ".".join(relative_path.parts[:-1])
        else:
            module_dot_path = ".".join(relative_path.parts[:-1]) + f".{relative_path.stem}" if relative_path.parts[:-1] else relative_path.stem
            
        package_map[str(file_path.resolve())] = module_dot_path
    return package_map

def parse_codebase_directory(root_dir: str | Path, output_json_path: str=settings.GRAPH_STORE_PATH) -> Dict[str, Any]:
    """
    Recursively processes an entire directory structure, resolves cross-module 
    dependencies, and aggregates the data into a single unified structural graph map.
    """
    root_path = Path(root_dir).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Provided path does not exist: {root_path}")

    # Pass 1: Build global project map for accurate relative/absolute package naming
    print("🔍 Step 1: Mapping project directory tree paths...")
    project_modules = build_package_map(root_path)
    
    master_graph = {
        "nodes": [],
        "relationships": []
    }
    
    processed_count = 0
    print(f"📦 Step 2: Processing {len(project_modules)} Python module frameworks...")

    # Pass 2: Extract details per file
    for file_str_path, absolute_module_name in project_modules.items():
        file_path = Path(file_str_path)
        print(f"  ⚡ Extracting structural maps for: {absolute_module_name}")
        
        try:
            # Initialize Extractor
            extractor = CodeGraphExtractor(file_path)
            
            # OVERRIDE single stem name with the globally accurate package layout position
            extractor.module_name = absolute_module_name
            
            # Execute AST analysis blocks
            file_graph = extractor.extract()
            
            if file_graph:
                master_graph["nodes"].extend(file_graph["nodes"])
                master_graph["relationships"].extend(file_graph["relationships"])
                processed_count += 1
                
        except Exception as e:
            print(f"  ❌ Error compiling file path target {file_path.name}: {str(e)}")
            continue

    # Step 3: Write out completed schemas
    output_path = Path(output_json_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(master_graph, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 Generation Complete! Processed {processed_count} files.")
    print(f"💾 Master graph schema saved safely to: {output_path.resolve()}")
    return master_graph

if __name__ == "__main__":
    base = Path(__file__).parent / "data"
    inp_directory = base / "repo"
    out_file_path = base / "code_graph.json"

    parse_codebase_directory(inp_directory, out_file_path)

