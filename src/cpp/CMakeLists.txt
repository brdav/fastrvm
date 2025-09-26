# Build the C++ core library for sparsebayes and its dependencies.
cmake_minimum_required(VERSION 3.18)

project(sparsebayes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# --- Armadillo vendoring using FetchContent ------------------------------
set(ARMADILLO_VERSION "15.0.2")
set(HEADER_ONLY ON CACHE BOOL "Build Armadillo header-only")

FetchContent_Declare(
  armadillo_vendored
  URL "https://sourceforge.net/projects/arma/files/armadillo-${ARMADILLO_VERSION}.tar.xz/download"
  URL_HASH SHA256=990ab4ccb7eff1b6d70409e9aa7fa4119877ac5f5d10ba219e98460ab3e4d6eb
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(armadillo_vendored)

# --- Ensmallen vendoring using FetchContent ------------------------------
set(ENSMALLEN_VERSION "2.22.2")

FetchContent_Declare(
  ensmallen_vendored
  URL "https://ensmallen.org/files/ensmallen-${ENSMALLEN_VERSION}.tar.gz"
  URL_HASH SHA256=da9ce4bdd07f2c8d950e3797456da3152dfa5d1b0f5987a489fbe224f52e7e4f
  DOWNLOAD_EXTRACT_TIMESTAMP true
  SOURCE_SUBDIR "PATH_THAT_DOES_NOT_EXIST" # Prevents automatic add_subdirectory
)
FetchContent_MakeAvailable(ensmallen_vendored)
add_library(ensmallen INTERFACE)
target_include_directories(ensmallen INTERFACE "${ensmallen_vendored_SOURCE_DIR}/include")

# --- Core library --------------------------------------------------------
add_library(sparsebayes STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse_bayes.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/alignment_helpers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/evaluate_helpers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/update_helpers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/action_helpers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/laplace.cpp
)
target_include_directories(sparsebayes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Ensure the static library is built as PIC so it can be linked into a shared module
set_target_properties(sparsebayes PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Link to external dependencies
target_link_libraries(sparsebayes PUBLIC armadillo)
target_link_libraries(sparsebayes PUBLIC ensmallen)
