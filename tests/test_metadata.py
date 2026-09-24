from importlib.metadata import version

import reciprocal


def test_runtime_version_comes_from_distribution_metadata():
    assert reciprocal.__version__ == version("reciprocal")
