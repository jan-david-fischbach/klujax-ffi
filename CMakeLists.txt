cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX C)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax.extend import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)

message(STATUS "XLA include directory: ${XLA_DIR}")
message(STATUS "We are located in: ${CMAKE_CURRENT_SOURCE_DIR}")

find_package(nanobind CONFIG REQUIRED)

file(
  GLOB_RECURSE SRCS_SUITSPARSE 
  suitesparse/SuiteSparse_config/*.c
  suitesparse/AMD/Source/*.c
  suitesparse/BTF/Source/*.c
  suitesparse/COLAMD/Source/*.c
  suitesparse/KLU/Source/*.c
)

set(
  HDRS_SUITSPARSE 
  suitesparse/SuiteSparse_config/
  suitesparse/AMD/Include/
  suitesparse/BTF/Include/
  suitesparse/COLAMD/Include/
  suitesparse/KLU/Include/
)

# message(STATUS "The following Suitsparse Headers were found: ${HDRS_SUITSPARSE}")
# message(STATUS "The following Suitsparse Sources were found: ${SRCS_SUITSPARSE}")

add_library(SuiteSparse ${SRCS_SUITSPARSE})
target_include_directories(SuiteSparse PRIVATE ${HDRS_SUITSPARSE})
set_source_files_properties(${SRCS_SUITSPARSE} PROPERTIES LANGUAGE C)
set_target_properties(SuiteSparse PROPERTIES LINKER_LANGUAGE C)
set_property(TARGET SuiteSparse PROPERTY POSITION_INDEPENDENT_CODE ON) 
# TODO figure out shared linking
install(TARGETS SuiteSparse LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

nanobind_add_module(_klujax NB_STATIC "src/klujax_ffi/klujax.cpp")
target_include_directories(_klujax PUBLIC ${XLA_DIR} build_dep)

target_link_libraries(_klujax PRIVATE SuiteSparse)
install(TARGETS _klujax LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

