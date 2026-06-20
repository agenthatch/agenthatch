"""Simulate Phase 1.5 ScriptAnalyzer on the data-analyzer test skill.

This script mirrors the analyzer.py design from v0.8.0 Part A.
It exercises: function extraction, class-method skipping, nested-function
skipping, async detection, docstring parsing, JSON Schema generation.
"""
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Models (from v0.8.0 Part A) ──

@dataclass
class FunctionSignature:
    name: str
    params: list[dict[str, Any]] = field(default_factory=list)
    return_type: str = "Any"
    docstring: str | None = None
    description: str | None = None
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)

    def to_json_schema(self) -> dict[str, Any]:
        properties = {}
        required = []
        for p in self.params:
            prop = {"description": p.get("description", ""), "type": _python_type_to_json(p.get("type", "Any"))}  # noqa: E501
            properties[p["name"]] = prop
            if not p.get("optional", False):
                required.append(p["name"])
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or self.docstring or "",
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }


@dataclass
class ScriptAnalysis:
    path: str
    language: str
    functions: list[FunctionSignature]
    parse_error: str | None = None


# ── Helpers (from v0.8.0 Part A) ──

_PYTHON_TO_JSON: dict[str, str] = {
    "str": "string", "int": "integer", "float": "number",
    "bool": "boolean", "list": "array", "dict": "object",
    "None": "null", "Any": "string",
}


def _python_type_to_json(type_name: str) -> str:
    type_name = type_name.strip()
    if type_name.startswith("Optional["):
        type_name = type_name[9:-1]
    return _PYTHON_TO_JSON.get(type_name, "string")


def _node_contains(parent: ast.AST, child: ast.AST) -> bool:
    parent_start = parent.lineno
    parent_end = getattr(parent, "end_lineno", parent_start)
    child_start = child.lineno
    return parent_start <= child_start <= parent_end


def _short_docstring(docstring: str | None) -> str | None:
    if not docstring:
        return None
    for line in docstring.strip().split("\n"):
        line = line.strip()
        if line:
            return line
    return docstring.strip()


def _parse_param_descriptions(docstring: str | None) -> dict[str, str]:
    if not docstring:
        return {}
    descs: dict[str, str] = {}
    import re
    for line in docstring.strip().split("\n"):
        m = re.match(r"\s*:param\s+(\w+)\s*:\s*(.*)", line.strip())
        if m:
            descs[m.group(1)] = m.group(2).strip()
    return descs


def _extract_type(annotation: ast.expr | None) -> str:
    if annotation is None:
        return "Any"
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Subscript):
        if isinstance(annotation.value, ast.Name):
            return f"{annotation.value.id}[...]"
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return "None"
    return "Any"


def _decorator_name(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return ""


def _extract_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionSignature | None:  # noqa: E501
    try:
        name = node.name
        docstring = ast.get_docstring(node)
        description = _short_docstring(docstring)
        param_descs = _parse_param_descriptions(docstring)

        params = []
        for arg in node.args.args:
            pname = arg.arg
            ptype = _extract_type(arg.annotation)
            params.append({
                "name": pname,
                "type": ptype,
                "description": param_descs.get(pname, ""),
                "optional": False,
            })

        return_type = _extract_type(node.returns)
        decorators = [_decorator_name(d) for d in node.decorator_list]
        is_async = isinstance(node, ast.AsyncFunctionDef)

        return FunctionSignature(
            name=name,
            params=params,
            return_type=return_type,
            docstring=docstring,
            description=description,
            is_async=is_async,
            decorators=decorators,
        )
    except Exception:
        return None


# ── Main Analysis (from v0.8.0 Part A) ──

def analyze_python_script(path: Path, rel_path: str) -> ScriptAnalysis:
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        return ScriptAnalysis(path=rel_path, language="python", functions=[], parse_error=str(e))

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return ScriptAnalysis(path=rel_path, language="python", functions=[], parse_error=str(e))

    analysis = ScriptAnalysis(path=rel_path, language="python", functions=[])

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue

            # Skip methods inside classes and nested functions (R-3 fix)
            is_nested = False
            for parent in ast.walk(tree):
                if parent is node:
                    continue
                if isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _node_contains(parent, node):
                        is_nested = True
                        break
            if is_nested:
                continue

            sig = _extract_function_signature(node)
            if sig:
                analysis.functions.append(sig)

    return analysis


# ── Test Execution ──

skill_dir = Path(__file__).parent / "fixtures" / "skills" / "data-analyzer"  # noqa: E501
py_file = skill_dir / "scripts" / "compute.py"

print("=" * 70)
print("Phase 1.5 Simulation: AST Analysis of compute.py")
print("=" * 70)

analysis = analyze_python_script(py_file, "scripts/compute.py")

print(f"\nFile: {analysis.path}")
print(f"Language: {analysis.language}")
print(f"Parse error: {analysis.parse_error}")
print(f"Functions extracted: {len(analysis.functions)}")

all_pass = True
for i, func in enumerate(analysis.functions):
    print(f"\n--- Function {i+1}: {func.name} ---")
    print(f"  is_async: {func.is_async}")
    print(f"  decorators: {func.decorators}")
    print(f"  return_type: {func.return_type}")
    print(f"  description: {func.description}")
    print("  params:")
    for p in func.params:
        print(f"    - {p['name']}: {p['type']} | {p['description']}")

    schema = func.to_json_schema()
    print("  JSON Schema:")
    print(f"    name: {schema['function']['name']}")
    print(f"    params: {list(schema['function']['parameters']['properties'].keys())}")
    print(f"    required: {schema['function']['parameters']['required']}")

# Expected results validation
extracted = {f.name: f for f in analysis.functions}

print(f"\n{'=' * 70}")
print("Validation")
print("=" * 70)

checks = []

# Check 1: compute_stats extracted
if "compute_stats" in extracted:
    f = extracted["compute_stats"]
    t1 = len(f.params) == 1
    t2 = not f.is_async
    checks.append((t1, f"compute_stats: {len(f.params)} params (expected 1)"))
    checks.append((t2, "compute_stats: is_async=False"))
    all_pass = all_pass and t1 and t2
else:
    checks.append((False, "compute_stats not extracted"))
    all_pass = False

# Check 2: detect_outliers extracted
if "detect_outliers" in extracted:
    f = extracted["detect_outliers"]
    t1 = len(f.params) == 2
    t2 = not f.is_async
    checks.append((t1, f"detect_outliers: {len(f.params)} params (expected 2)"))
    checks.append((t2, "detect_outliers: is_async=False"))
    all_pass = all_pass and t1 and t2
else:
    checks.append((False, "detect_outliers not extracted"))
    all_pass = False

# Check 3: load_csv extracted as async
if "load_csv" in extracted:
    f = extracted["load_csv"]
    t1 = len(f.params) == 2
    t2 = f.is_async
    checks.append((t1, f"load_csv: {len(f.params)} params (expected 2)"))
    checks.append((t2, "load_csv: is_async=True"))
    all_pass = all_pass and t1 and t2
else:
    checks.append((False, "load_csv not extracted"))
    all_pass = False

# Check 4: _validate_numeric NOT extracted
if "_validate_numeric" in extracted:
    checks.append((False, "_validate_numeric extracted (should be skipped)"))
    all_pass = False
else:
    checks.append((True, "_validate_numeric skipped (private function)"))

# Check 5: Correct count
if len(analysis.functions) == 3:
    checks.append((True, f"Exactly 3 top-level functions ({len(analysis.functions)} extracted)"))
else:
    checks.append((False, f"Expected 3 functions, got {len(analysis.functions)}"))
    all_pass = False

for passed, msg in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {msg}")

print(f"\n{'=' * 70}")
print(f"OVERALL: {'ALL PASSED' if all_pass else 'SOME FAILURES'}")
print("=" * 70)

sys.exit(0 if all_pass else 1)
