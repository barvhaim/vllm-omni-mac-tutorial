from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_complete_beginner_curriculum_is_present():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 13
    assert notebooks[0].name == "00_course_orientation.ipynb"
    assert notebooks[-1].name == "12_deployment_and_graduation.ipynb"


def test_every_notebook_has_learning_goals_and_mac_boundary():
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        nb = nbformat.read(path, as_version=4)
        text = "\n".join("".join(cell.source) for cell in nb.cells if cell.cell_type == "markdown")
        assert "Learning goals" in text, path.name
        assert "Mac track" in text, path.name
