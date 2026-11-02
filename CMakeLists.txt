cmake_minimum_required(VERSION 3.28)
project(anndata VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18)
  message(FATAL_ERROR "The initial supported toolchain is Clang 18 or newer. Configure with -DCMAKE_CXX_COMPILER=clang++-18 and Ninja.")
endif()
if(NOT CMAKE_GENERATOR MATCHES "Ninja")
  message(FATAL_ERROR "Use Ninja (>=1.11) for C++20 module dependency scanning.")
endif()
if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
  string(REGEX MATCH "^[0-9]+" _anndata_clang_major "${CMAKE_CXX_COMPILER_VERSION}")
  find_program(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS
    NAMES clang-scan-deps-${_anndata_clang_major} clang-scan-deps
    REQUIRED)
endif()

option(ANNDATA_BUILD_TESTS "Build standalone native tests" ${PROJECT_IS_TOP_LEVEL})
option(ANNDATA_BUILD_EXAMPLES "Build import-only examples" ${PROJECT_IS_TOP_LEVEL})
option(ANNDATA_BUILD_BENCHMARKS "Build numerical memory benchmarks" OFF)
option(ANNDATA_ENABLE_INTEROP_TESTS "Compare semantics using Python AnnData fixtures" OFF)
option(ANNDATA_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

add_library(anndata)
add_library(anndata::anndata ALIAS anndata)
target_compile_features(anndata PUBLIC cxx_std_20)
set_target_properties(anndata PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_sources(anndata
  PUBLIC FILE_SET CXX_MODULES BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" FILES
    src/anndata.cppm
    src/anndata.core.cppm
    src/anndata.arrays.cppm
    src/anndata.data.cppm
  PRIVATE src/arrays.cpp src/data.cpp)
target_compile_options(anndata PRIVATE -Wall -Wextra -Wpedantic)
if(ANNDATA_ENABLE_SANITIZERS)
  target_compile_options(anndata PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(anndata PUBLIC -fsanitize=address,undefined)
endif()

install(TARGETS anndata EXPORT anndataTargets
  FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/anndata/modules
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT anndataTargets NAMESPACE anndata:: CXX_MODULES_DIRECTORY cxx-modules
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/anndata)
configure_package_config_file(cmake/anndataConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/anndataConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/anndata)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/anndataConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/anndataConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/anndataConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/anndata)
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/anndata)

if(ANNDATA_BUILD_EXAMPLES)
  add_executable(anndata_example examples/foundation.cpp)
  target_link_libraries(anndata_example PRIVATE anndata::anndata)
endif()
if(ANNDATA_BUILD_TESTS)
  enable_testing()
  foreach(_suite arrays data)
    add_executable(test_${_suite} tests/test_${_suite}.cpp)
    target_link_libraries(test_${_suite} PRIVATE anndata::anndata)
    add_test(NAME ${_suite} COMMAND test_${_suite})
  endforeach()
  if(ANNDATA_BUILD_EXAMPLES)
    add_test(NAME import_only_example COMMAND anndata_example)
  endif()
  add_test(NAME installed_consumer COMMAND ${CMAKE_COMMAND}
    -DANNDATA_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DANNDATA_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DANNDATA_COMPILER=${CMAKE_CXX_COMPILER}
    -DANNDATA_CONFIG=$<CONFIG>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test-installed-consumer.cmake)
endif()
if(ANNDATA_ENABLE_INTEROP_TESTS)
  enable_testing()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_executable(interop_driver tests/interop_driver.cpp)
  target_link_libraries(interop_driver PRIVATE anndata::anndata)
  add_test(NAME python_semantics COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_semantics.py $<TARGET_FILE:interop_driver>)
endif()
if(ANNDATA_BUILD_BENCHMARKS)
  find_package(Threads REQUIRED)
  add_executable(memory_benchmark benchmarks/memory.cpp)
  target_link_libraries(memory_benchmark PRIVATE anndata::anndata Threads::Threads)
endif()
