SRC_DIR = build
OBJ_DIR = build/obj
IDIR=$(SRC_DIR)
CC=g++
CPPFLAGS=-I$(IDIR) -L$(OBJ_DIR) -g

SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC_FILES))

test:
	pytest tests.py

suitesparse:
	git clone --depth 1 --branch v7.2.0 https://github.com/DrTimothyAldenDavis/SuiteSparse suitesparse || true
	cd suitesparse && rm -rf .git

build:
	mkdir $(SRC_DIR)
	mkdir $(OBJ_DIR)
	cp suitesparse/SuiteSparse_config/*.h build 
	cp suitesparse/SuiteSparse_config/*.h build
	cp suitesparse/SuiteSparse_config/*.c build
	cp suitesparse/AMD/Include/*.h build
	cp suitesparse/AMD/Source/*.c build
	cp suitesparse/BTF/Include/*.h build
	cp suitesparse/BTF/Source/*.c build
	cp suitesparse/COLAMD/Include/*.h build
	cp suitesparse/COLAMD/Source/*.c build
	cp suitesparse/KLU/Include/*.h build
	cp suitesparse/KLU/Source/*.c build
	cp suitesparse/LICENSE.txt build
	cp klujax_ffi/klujax.cpp build


klujax.out: $(OBJ_FILES) klujax_ffi/klujax.cpp
	$(CC) -o klujax.out $(CPPFLAGS) klujax_ffi/klujax.cpp $(OBJ_FILES)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) -c -o $@ $<

.PHONY: clean
.PHONY: all

clean:
	find . -not -path "./suitesparse*" -name "dist" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "build" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "builds" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "__pycache__" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "*.so" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "*.egg-info" | xargs rm -rf
	find . -not -path "./suitesparse*" -name ".ipynb_checkpoints" | xargs rm -rf
	find . -not -path "./suitesparse*" -name ".pytest_cache" | xargs rm -rf

