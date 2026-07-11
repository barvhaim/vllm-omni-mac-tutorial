.PHONY: test notebooks
notebooks:
	uv run python scripts/execute_notebooks.py

test:
	uv run pytest -q
	uv run python scripts/execute_notebooks.py
