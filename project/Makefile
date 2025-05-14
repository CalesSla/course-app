.PHONY: run_builder run_inference install clean check runner_builder runner_inference
.DEFAULT_GOAL:=runner_inference

run_builder:
	cd src; poetry run python runner_builder.py

run_inference:
	cd src; poetry run python runner_inference.py

install: pyproject.toml
	poetry install

clean:
	rm -rf `find . -type d -name __pycache__`

check:
	poetry run flake8 src/

runner_builder: run_builder clean

runner_inference: run_inference clean