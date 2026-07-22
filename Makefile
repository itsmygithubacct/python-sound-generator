.PHONY: test lint doctor verify build clean

test:
	python3 -m pytest

lint:
	python3 -m ruff check .

doctor:
	python3 -m python_sound_generator doctor

verify: lint test doctor

build:
	python3 -m build

clean:
	rm -rf build dist .pytest_cache .ruff_cache python_sound_generator.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
