.PHONY: help install dev test lint format train serve clean docker-build

help:
	@echo "nano-gpt-prod Development Tasks:"
	@echo "  install      Install production dependencies and package"
	@echo "  dev          Install development and test dependencies"
	@echo "  test         Run complete unit and integration test suite"
	@echo "  lint         Run code linting with flake8"
	@echo "  format       Format code with black and isort"
	@echo "  train        Run model training on Shakespeare"
	@echo "  serve        Start API server and Web Studio"
	@echo "  clean        Remove temporary build artifacts and pycache"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	python -m unittest discover tests -v

lint:
	flake8 gpt tests --max-line-length=100

format:
	black gpt tests *.py
	isort gpt tests *.py

train:
	python train.py --config configs/shakespeare_char.yaml

serve:
	python serve.py --open

docker-build:
	docker build -t nano-gpt-prod:latest .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info .pytest_cache
