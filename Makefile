# Make targets for the project

.PHONY: install format

REQ_FILE=requirements.txt

install:
	@echo "Installing uv..."
	python -m pip install --upgrade pip uv
	@echo "Installing project dependencies with uv..."
	uv pip install -r $(REQ_FILE)

format:
	@echo "Installing formatting tools (black, isort)..."
	python -m pip install --upgrade black isort >/dev/null
	@echo "Running isort..."
	isort .
	@echo

