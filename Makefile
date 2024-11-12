test:
	python tests/klujax_test.py

suitesparse:
	git clone --depth 1 --branch v7.2.0 https://github.com/DrTimothyAldenDavis/SuiteSparse suitesparse || true
	cd suitesparse && rm -rf .git

.PHONY: clean

clean:
	find . -not -path "./suitesparse*" -name "dist" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "build" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "builds" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "__pycache__" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "*.so" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "*.egg-info" | xargs rm -rf
	find . -not -path "./suitesparse*" -name ".ipynb_checkpoints" | xargs rm -rf
	find . -not -path "./suitesparse*" -name ".pytest_cache" | xargs rm -rf

