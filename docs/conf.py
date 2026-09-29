"""Sphinx configuration for the reciprocal documentation."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "docs" / "_build" / "matplotlib"))

project = "reciprocal"
author = "Phillip Manley"
copyright = "2026, Phillip Manley"

try:
    release = version("reciprocal")
except PackageNotFoundError:
    release = "1.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True

templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
html_title = f"reciprocal {release}"

plot_apply_rcparams = True
plot_formats = [("png", 120)]
plot_html_show_source_link = False
plot_rcparams = {
    "axes.grid": False,
    "figure.figsize": (6.0, 5.0),
    "figure.constrained_layout.use": True,
    "savefig.bbox": "tight",
}
