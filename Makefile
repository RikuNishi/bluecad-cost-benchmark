.PHONY: check test env

check: env test

env:
	python3 scripts/check_env.py

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v
