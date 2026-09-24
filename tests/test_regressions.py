import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_symmetry_alias_factory_is_defined_once():
    tree = ast.parse((ROOT / "src/reciprocal/symmetry.py").read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "symmetry_from_alias"
    ]
    assert len(definitions) == 1


def test_experimental_stack_uses_real_value_error():
    source = (ROOT / "src/kspace_sample/StackSystem.py").read_text()
    assert "ValueEror" not in source
    assert 'raise ValueError("cannot search for value 1")' in source


def test_supported_package_has_no_assert_based_input_validation():
    for path in (ROOT / "src/reciprocal").glob("*.py"):
        tree = ast.parse(path.read_text())
        assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree)), path
