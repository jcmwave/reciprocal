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


def test_experimental_package_tree_is_not_distributed():
    """The unsupported historical package was deliberately retired in P1."""
    assert not (ROOT / "src/kspace_sample").exists()
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'include = ["reciprocal*"]' in pyproject


def test_supported_package_has_no_assert_based_input_validation():
    for path in (ROOT / "src/reciprocal").glob("*.py"):
        tree = ast.parse(path.read_text())
        assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree)), path
