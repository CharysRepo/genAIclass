import ast
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional, Union

@dataclass
class CodeNode:
    """A structured representation of a code graph node with standardized properties."""
    node_id: str
    node_type: str
    name: str
    file_path: str
    lineno: int
    end_lineno: Optional[int] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    code_snippet: str = ""

@dataclass
class CodeRelationship:
    """A clean container for a code graph relationship without line number metadata."""
    source: str
    target: str
    rel_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


class CodeGraphExtractor:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.module_name: str = file_path.stem
        self.source_code: str = ""
        
        # Scoping states
        self.current_class: str | None = None
        self.current_caller: str | None = None
        
        # Type Stores
        self.nodes: Dict[str, CodeNode] = {}
        self.relationships: List[CodeRelationship] = []
        
        # Track explicit dependency types (standard vs third_party/local)
        self.module_dependencies: Dict[str, str] = {}

        # Track imported names
        self.imported_names: Dict[str, str] = {}  

    def _get_decorators(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
        """Helper to extract and resolve absolute decorator names applied to a class or function."""
        decorators = []
        
        def _resolve_name(expr: ast.AST) -> str | None:
            """Recursively flattens ast structural paths into standard dot-notation strings."""
            if isinstance(expr, ast.Name):
                return expr.id
            elif isinstance(expr, ast.Attribute):
                prefix = _resolve_name(expr.value)
                return f"{prefix}.{expr.attr}" if prefix else expr.attr
            elif isinstance(expr, ast.Call):
                return _resolve_name(expr.func)
            return None

        for dec in node.decorator_list:
            raw_path = _resolve_name(dec)
            if not raw_path:
                continue
                
            # Cross-Module Resolution Strategy
            base_identifier = raw_path.split('.')[0]
            if base_identifier in self.imported_names:
                # Scenario A: The root object or function was explicitly imported
                resolved_base = self.imported_names[base_identifier]
                if '.' in raw_path:
                    # Append remaining attributes (e.g., if 'api' -> 'core.api', then 'api.route' -> 'core.api.route')
                    remainder = '.'.join(raw_path.split('.')[1:])
                    resolved_path = f"{resolved_base}.{remainder}"
                else:
                    resolved_path = resolved_base
            else:
                # Scenario B: It's a local decorator defined inside this exact module file
                resolved_path = f"{self.module_name}.{raw_path}"
                
            decorators.append(resolved_path)
            
        return decorators


    def _add_node(self, node_id: str, node_type: str, name: str, ast_node: ast.AST, extra_props: Dict[str, Any] | None = None):
        """Helper to fully construct and register a CodeNode model object, filling all core fields."""
        lineno = getattr(ast_node, 'lineno', 1)
        end_lineno = getattr(ast_node, 'end_lineno', None)
        if isinstance(ast_node, ast.Module) and self.source_code:
            end_lineno = len(self.source_code.splitlines())
            
        snippet = ast.get_source_segment(self.source_code, ast_node) or ""

        base_props = {}
        docstring = ast.get_docstring(ast_node) if isinstance(ast_node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else None
        if docstring:
            base_props["docstring"] = docstring

        final_props = {**base_props, **(extra_props or {})}

        self.nodes[node_id] = CodeNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            file_path=str(self.file_path),
            lineno=lineno,
            end_lineno=end_lineno,
            properties=final_props,
            code_snippet=snippet.strip()
        )

    def _add_relationship(self, source: str, rel_type: str, target: str, properties: Dict[str, Any] | None = None):
        """Creates and appends a structured CodeRelationship object with dynamic properties."""
        final_props = properties or {}
        self.relationships.append(
            CodeRelationship(source=source, target=target, rel_type=rel_type, properties=final_props)
        )

    def _process_import(self, node: ast.Import | ast.ImportFrom):
        """Processes imported names and classifies them as standard, third_party, or local."""
        # 1. Gather all built-in and standard library names
        stdlib_names = getattr(sys, "stdlib_module_names", set())
        builtin_names = set(sys.builtin_module_names)
        standard_fixtures = stdlib_names.union(builtin_names)

        # 2. Identify local modules in the directory relative to the file being processed
        local_dir = self.file_path.parent
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split('.')[0]
                
                # Classification Logic
                if base_module in standard_fixtures:
                    dep_type = "standard"
                elif (local_dir / f"{base_module}.py").exists() or (local_dir / base_module).is_dir():
                    dep_type = "local"
                else:
                    dep_type = "third_party"

                self.module_dependencies[base_module] = dep_type
                
                # Track the import alias for resolution (e.g., import math as m -> m points to math)
                local_name = alias.asname if alias.asname else alias.name
                self.imported_names[local_name] = alias.name

                self._add_node(node_id=alias.name, node_type="Module", name=alias.name, ast_node=node)
                self._add_relationship(
                    source=self.module_name, 
                    rel_type="imports", 
                    target=alias.name, 
                    properties={"type": dep_type}
                )
                
        elif isinstance(node, ast.ImportFrom) and node.module:
            base_module = node.module.split('.')[0]
            
            # Relative import check (e.g., from .utils import foo)
            if node.level > 0:
                dep_type = "local"
            elif base_module in standard_fixtures:
                dep_type = "standard"
            elif (local_dir / f"{base_module}.py").exists() or (local_dir / base_module).is_dir():
                dep_type = "local"
            else:
                dep_type = "third_party"

            self.module_dependencies[base_module] = dep_type
            
            # Handle imported items (e.g., from module import function_a, ClassB)
            for alias in node.names:
                local_name = alias.asname if alias.asname else alias.name
                # Store absolute reference: "module_name.imported_item"
                full_import_path = f"{node.module}.{alias.name}"
                self.imported_names[local_name] = full_import_path

            self._add_node(node_id=node.module, node_type="Module", name=node.module, ast_node=node)
            self._add_relationship(
                source=self.module_name, 
                rel_type="imports", 
                target=node.module, 
                properties={"type": dep_type}
            )


    def _process_class(self, node: ast.ClassDef):
        # Create a fully qualified class name for unique node IDs
        class_name = f"{self.module_name}.{node.name}"
        
        # Extract parent classes and handle inheritance links
        base_classes = []
        for base in node.bases:
            # 1. Resolve base class name
            if isinstance(base, ast.Name):
                base_raw = base.id
            elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                base_raw = f"{base.value.id}.{base.attr}"
            else:
                base_raw = base.attr if hasattr(base, 'attr') else None

            if base_raw:
                # 2. Check if the base class was imported from another module
                if base_raw in self.imported_names:
                    resolved_base = self.imported_names[base_raw]
                elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id in self.imported_names:
                    resolved_base = f"{self.imported_names[base.value.id]}.{base.attr}"
                else:
                    # Fallback to local module scope
                    resolved_base = f"{self.module_name}.{base_raw}"
                
                base_classes.append(resolved_base)
                self._add_relationship(class_name, "extends", resolved_base)

        class_props = {
            "extends": base_classes,
            "decorators": self._get_decorators(node),
            "is_nested": self.current_class is not None,
            "instance_attributes": []
        }

        self._add_node(
            node_id=class_name, 
            node_type="Class", 
            name=node.name, 
            ast_node=node, 
            extra_props=class_props
        )
        self._add_relationship(self.module_name, "defines", class_name)
        
        # Maintain fully qualified scoping state for tracking nested members
        old_class = self.current_class
        self.current_class = class_name
        for child in ast.iter_child_nodes(node):
            self.parse_ast(child)
        self.current_class = old_class

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        args_list = [arg.arg for arg in node.args.args]
        arg_count = len(args_list)
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        func_props = {
            "argument_count": arg_count,
            "arguments_list": args_list,
            "is_async": is_async,
            "decorators": self._get_decorators(node),
            "is_nested_function": self.current_caller is not None
        }
        
        if self.current_class:
            # unique_name is already fully qualified because self.current_class contains the module prefix
            unique_name = f"{self.current_class}.{node.name}"
            node_type = "AsyncMethod" if is_async else "Method"
            func_props["belongs_to"] = self.current_class
            
            self._add_node(node_id=unique_name, node_type=node_type, name=node.name, ast_node=node, extra_props=func_props)
            self._add_relationship(self.current_class, "defines", unique_name)
            active_caller = unique_name
        else:
            # Top-level standalone function
            unique_name = f"{self.module_name}.{node.name}"
            node_type = "AsyncFunction" if is_async else "Function"
            
            self._add_node(node_id=unique_name, node_type=node_type, name=node.name, ast_node=node, extra_props=func_props)
            self._add_relationship(self.module_name, "defines", unique_name)
            active_caller = unique_name

        old_caller = self.current_caller
        self.current_caller = active_caller
        for child in ast.iter_child_nodes(node):
            self.parse_ast(child)
        self.current_caller = old_caller

    def _process_call(self, node: ast.Call):
        if not self.current_caller:
            return

        called_name = None
        is_external = False

        # Case 1: Direct function or class call (e.g., calculate_metrics())
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.imported_names:
                called_name = self.imported_names[func_name]
                is_external = True
            else:
                # Local call within the same file module
                called_name = f"{self.module_name}.{func_name}"

        # Case 2: Attribute access call (e.g., utils.calculate_metrics() or self.logger.info())
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                attr_name = node.func.attr
                
                # Check if the root identifier was explicitly imported (e.g., import utils)
                if obj_name in self.imported_names:
                    called_name = f"{self.imported_names[obj_name]}.{attr_name}"
                    is_external = True
                elif obj_name == "self" and self.current_class:
                    # Point method calls back to the owning class scope
                    called_name = f"{self.current_class}.{attr_name}"
                else:
                    # Fallback for generic objects or local instances
                    called_name = f"{self.module_name}.{obj_name}.{attr_name}"
            else:
                # Nested structures (e.g., a.b.c()) fallback to terminal name
                called_name = node.func.attr

        if called_name:
            rel_type = "calls_external" if is_external else "calls"
            self._add_relationship(
                source=self.current_caller, 
                rel_type=rel_type, 
                target=called_name,
                properties={"is_external": is_external}
            )

    def _process_assignment(self, node: Union[ast.Assign, ast.AnnAssign, ast.AugAssign]):
        """Extracts variable names from assignment expressions and constructs scope-based nodes."""

        # --- Instance Attributes Detection Block ---
        if self.current_class and self.current_caller:
            instance_targets = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                        instance_targets.append(t.attr)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                t = node.target
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                    instance_targets.append(t.attr)
                    
            if instance_targets:
                class_node = self.nodes.get(self.current_class)
                if class_node and "instance_attributes" in class_node.properties:
                    for attr in instance_targets:
                        if attr not in class_node.properties["instance_attributes"]:
                            class_node.properties["instance_attributes"].append(attr)
                return  # Skip processing instance variables as general local nodes

        # --- Standard Scoped Variables Mapping Logic ---
        targets = []
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    targets.append(t.id)
                elif isinstance(t, ast.Tuple) or isinstance(t, ast.List):
                    # Unpacking e.g., x, y = 1, 2
                    targets.extend([elt.id for elt in t.elts if isinstance(elt, ast.Name)])
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                targets.append(node.target.id)

        for var_name in targets:
            # Context Scenario 1: Inside a function/method -> Local Variable
            if self.current_caller:
                unique_id = f"{self.current_caller}.{var_name}"
                node_type = "LocalVariable"
                source_owner = self.current_caller
                
            # Context Scenario 2: Inside a class, outside a function -> Class Variable
            elif self.current_class:
                unique_id = f"{self.current_class}.{var_name}"
                node_type = "ClassVariable"
                source_owner = self.current_class
                
            # Context Scenario 3: Outside everything -> Global Module Variable
            else:
                unique_id = f"{self.module_name}.{var_name}"
                node_type = "GlobalVariable"
                source_owner = self.module_name

            # Avoid re-registering variable nodes if encountered multiple times
            if unique_id not in self.nodes:
                self._add_node(node_id=unique_id, node_type=node_type, name=var_name, ast_node=node)
                self._add_relationship(source=source_owner, rel_type="defines_variable", target=unique_id)


    def parse_ast(self, node: ast.AST):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._process_import(node)
        elif isinstance(node, ast.ClassDef):
            self._process_class(node)
            return
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._process_function(node)
            return
        elif isinstance(node, ast.Call):
            self._process_call(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            self._process_assignment(node)

        for child in ast.iter_child_nodes(node):
            self.parse_ast(child)

    def extract(self) -> Dict[str, Any]:
        try:
            self.source_code = self.file_path.read_text(encoding="utf-8")
            syntax_tree = ast.parse(self.source_code, filename=self.file_path.name)
        except SyntaxError as e:
            print(f"Syntax Error: {e}")
            return

        module_props = {
            "total_lines": len(self.source_code.splitlines()),
            "dependencies": self.module_dependencies
        }
        
        self._add_node(node_id=self.module_name, node_type="Module", name=self.module_name, ast_node=syntax_tree, extra_props=module_props)
        self.parse_ast(syntax_tree)

        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "relationships": [asdict(r) for r in self.relationships]
        }
    
if __name__ == "__main__":
    base = Path(__file__).parent / "data"
    inp_file_path = base / "sample3.py"
    out_file_path = base / "code_graph.json"

    graph_extractor = CodeGraphExtractor(inp_file_path)
    graph = graph_extractor.extract()

    with open(out_file_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=4, ensure_ascii=False)
        