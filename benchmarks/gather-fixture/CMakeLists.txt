cmake_minimum_required(VERSION 3.28)
project(anndatapp_gather_fixture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Gather fixture build type" FORCE)
endif()
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(FATAL_ERROR "Configure the fixture with -DCMAKE_BUILD_TYPE=Release")
endif()

if(NOT ANNDATAPP_SOURCE OR NOT IS_ABSOLUTE "${ANNDATAPP_SOURCE}" OR NOT IS_DIRECTORY "${ANNDATAPP_SOURCE}")
  message(FATAL_ERROR "Pass -DANNDATAPP_SOURCE=/absolute/path/to/frozen/source")
endif()
if(NOT ANNDATAPP_PROBE OR NOT IS_ABSOLUTE "${ANNDATAPP_PROBE}" OR NOT EXISTS "${ANNDATAPP_PROBE}")
  message(FATAL_ERROR "Pass -DANNDATAPP_PROBE=/absolute/path/to/benchmarks/gather.cpp")
endif()
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18)
  message(FATAL_ERROR "Use Clang 18 or newer for C++20 module dependency scanning")
endif()
if(NOT CMAKE_GENERATOR STREQUAL "Ninja")
  message(FATAL_ERROR "Use single-config Ninja for C++20 module dependency scanning")
endif()
if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
  string(REGEX MATCH "^[0-9]+" _clang_major "${CMAKE_CXX_COMPILER_VERSION}")
  find_program(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS
    NAMES clang-scan-deps-${_clang_major} clang-scan-deps REQUIRED)
endif()

# Keep module, consumer, and probe optimization settings identical in paired builds.
add_compile_options(-O3)

add_library(anndatapp_core STATIC)
target_sources(anndatapp_core
  PUBLIC FILE_SET CXX_MODULES BASE_DIRS "${ANNDATAPP_SOURCE}/src" FILES
    "${ANNDATAPP_SOURCE}/src/anndatapp.core.cppm"
    "${ANNDATAPP_SOURCE}/src/anndatapp.arrays.cppm"
    "${ANNDATAPP_SOURCE}/src/anndatapp.strings.cppm"
    "${ANNDATAPP_SOURCE}/src/anndatapp.data.cppm"
  PRIVATE
    "${ANNDATAPP_SOURCE}/src/arrays.cpp"
    "${ANNDATAPP_SOURCE}/src/strings.cpp"
    "${ANNDATAPP_SOURCE}/src/data.cpp")

add_executable(gather_probe "${ANNDATAPP_PROBE}")
target_link_libraries(gather_probe PRIVATE anndatapp_core)

add_executable(gather_consumer consumer.cpp)
target_link_libraries(gather_consumer PRIVATE anndatapp_core)
