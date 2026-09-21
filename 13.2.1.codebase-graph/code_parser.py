import ast
from pathlib import Path
import ast

def parse_ast(node: ast.AST, indent_level: int):
    indent = "  " * indent_level
    node_name = type(node).__name__        
    extra_info = ""
    
    if isinstance(node, ast.Name):
        extra_info = f" (id='{node.id}')"
    elif isinstance(node, ast.FunctionDef):
        extra_info = f" (name='{node.name}')"
    elif isinstance(node, ast.ClassDef):
        extra_info = f" (name='{node.name}')"
    elif isinstance(node, ast.Constant):
        extra_info = f" (value={repr(node.value)})"
        
    print(f"{indent}└── {node_name}{extra_info}")

    for child in ast.iter_child_nodes(node): 
        parse_ast(child, indent_level + 1)

def parse_python_file(file_path: Path):
    try:
        source_code = file_path.read_text(encoding="utf-8")
        root_ast = ast.parse(source_code, filename=file_path.name)        
        print(f"=== Abstract Syntax Tree for {file_path.name} ===\n")
        parse_ast(root_ast, 0)        
    except SyntaxError as e:
        print(f"Syntax Error in target file: {e}")


if __name__ == "__main__":
    file_path = Path(__file__).parent / "data/sample2.py"
    parse_python_file(file_path)
