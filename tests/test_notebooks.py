"""Notebook hygiene.

The notebooks are the deliverable a reader opens first. These checks are static,
on what is committed: each was executed top to bottom without an error, imports
from the package rather than reaching around it, and its stored output belongs to
the date it pins.
"""

import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def code_cells(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def text_output(cell):
    return "".join("".join(o.get("text", [])) for o in cell.get("outputs", [])
                   if o.get("output_type") == "stream")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
class TestEveryNotebook:
    def test_ran_without_errors(self, path):
        for cell in code_cells(load(path)):
            assert all(o.get("output_type") != "error" for o in cell.get("outputs", []))

    def test_was_executed_top_to_bottom_in_order(self, path):
        counts = [cell["execution_count"] for cell in code_cells(load(path))]
        assert counts == list(range(1, len(counts) + 1))

    def test_imports_the_package_instead_of_patching_the_path(self, path):
        source = "\n".join("".join(cell["source"]) for cell in code_cells(load(path)))
        assert "sys.path" not in source
        assert "from src." not in source and "import src" not in source

    def test_was_run_on_a_supported_python(self, path):
        version = load(path)["metadata"]["language_info"]["version"]
        assert version.startswith(("3.12.", "3.13."))


def test_the_first_notebooks_stored_output_belongs_to_the_date_it_pins():
    notebook = load(NOTEBOOKS[0])
    source = "\n".join("".join(cell["source"]) for cell in code_cells(notebook))
    pinned = re.search(r'AS_OF = "(\d{4}-\d{2}-\d{2})"', source)
    assert pinned, "the first notebook must pin AS_OF"
    output = "\n".join(text_output(cell) for cell in code_cells(notebook))
    assert f"to {pinned.group(1)}" in output
