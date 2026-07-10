.PHONY: test notebooks
notebooks:
	python scripts/execute_notebooks.py

test:
	pytest -q
	python scripts/execute_notebooks.py
