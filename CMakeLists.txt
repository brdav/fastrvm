# Build the Python extension for the fastrvm package.
cmake_minimum_required(VERSION 3.18)
project(fastrvm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build the C++ core library
add_subdirectory(src/cpp)

# Ensure Python interpreter/dev and pybind11 are available.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# The C++ core library 'sparsebayes' is expected to exist (built by src/cpp).
# Link against it.
pybind11_add_module(_sparsebayes_bindings ${CMAKE_SOURCE_DIR}/src/python/fastrvm/_sparsebayes_bindings.cpp)
target_link_libraries(_sparsebayes_bindings PRIVATE sparsebayes)

# Install into the Python package directory so scikit-build maps it into the wheel
install(TARGETS _sparsebayes_bindings DESTINATION fastrvm)
