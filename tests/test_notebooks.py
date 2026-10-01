import glob
import json

NOTEBOOKS = NOTEBOOKS = glob.glob("notebooks/*.ipynb")

def test_notebooks_are_valid_json():
    for path in NOTEBOOKS:
        with open(path) as f:
            json.load(f)


def test_no_committed_cell_outputs():
    # Committed outputs bloat diffs and can leak whatever ran last (including API responses).
    for path in NOTEBOOKS:
        with open(path) as f:
            nb = json.load(f)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                assert not cell.get("outputs"), f"{path} has a code cell with committed output"
                assert cell.get("execution_count") is None, f"{path} has a committed execution_count"


def test_no_api_key_pasted_into_a_cell():
    for path in NOTEBOOKS:
        with open(path) as f:
            text = f.read()
        assert "AIza" not in text, f"{path} looks like it has a literal Gemini key pasted in"