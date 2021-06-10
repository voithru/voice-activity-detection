all: dev-env test check
dev-env: set-commit-template install-pre-commit install-pytest install-requirements

clean: clean-pyc clean-test

# Development environment
set-commit-template:
	git config --local commit.template .gitmessage.txt

install-pre-commit:
	pip3 install --no-cache-dir pre-commit==2.12.1
	pre-commit install

install-pytest:
	pip3 install --no-cache-dir pytest==6.2.3 pytest-cov==2.11.1 pytest_xdist==2.2.1

install-requirements:
	pip3 install -r requirements.txt

# Test
test:
	python -m pytest --cov=vad tests/

# Check
check:
	pre-commit run -a

# Clean
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache
