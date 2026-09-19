cmake_minimum_required(VERSION 3.18)
project(usx LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(USX_BUILD_PYTHON "Build the Python extension module" ON)
option(USX_BUILD_TESTS  "Build the C++ test binaries" ON)
option(USX_NATIVE_ARCH  "Compile for the build machine's CPU (-march=native)" OFF)

# -----------------------------------------------------------------------------
# Core library
# -----------------------------------------------------------------------------
add_library(usx_core
  core/src/types.cpp
  core/src/geometry.cpp
  core/src/simulator.cpp
  core/src/demodulate.cpp
  core/src/beamformer.cpp
  core/src/postprocess.cpp
  core/src/doppler.cpp
  core/src/pipeline.cpp
)
target_include_directories(usx_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core/include>
  $<INSTALL_INTERFACE:include>
)

if(MSVC)
  target_compile_options(usx_core PRIVATE /O2 /W4)
else()
  target_compile_options(usx_core PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)
  # -ffast-math is deliberately NOT used: the beamformer's phase arithmetic and
  # the Cholesky solve in the minimum-variance combiner both depend on IEEE
  # semantics, and silent reassociation there produces wrong images rather than
  # slow ones.
endif()

if(USX_NATIVE_ARCH AND NOT MSVC)
  target_compile_options(usx_core PRIVATE -march=native)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(usx_core PUBLIC OpenMP::OpenMP_CXX)
  message(STATUS "usx: OpenMP enabled")
else()
  message(STATUS "usx: OpenMP not found — building single-threaded")
endif()

# -----------------------------------------------------------------------------
# Python extension
# -----------------------------------------------------------------------------
if(USX_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_FOUND_RESULT
  )
  if(NOT PYBIND11_FOUND_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 not found. pip install pybind11")
  endif()
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_usx bindings/pymodule.cpp)
  target_link_libraries(_usx PRIVATE usx_core)
  # The trailing `$<0:>` is an empty generator expression, and it matters: on
  # multi-config generators (Visual Studio, Xcode) CMake otherwise appends
  # `/$<CONFIG>` to LIBRARY_OUTPUT_DIRECTORY, so the module lands in
  # `python/usx/Release/` and neither the editable install nor
  # `wheel.packages = ["python/usx"]` can find it. Naming a generator
  # expression suppresses that per-config suffix.
  set_target_properties(_usx PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/usx$<0:>"
  )
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(USX_BUILD_TESTS)
  enable_testing()
  add_executable(usx_tests core/tests/test_main.cpp)
  target_link_libraries(usx_tests PRIVATE usx_core)
  add_test(NAME usx_tests COMMAND usx_tests)

  add_executable(usx_bench core/tests/benchmark.cpp)
  target_link_libraries(usx_bench PRIVATE usx_core)
endif()
