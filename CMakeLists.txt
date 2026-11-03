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
option(ANNDATA_ENABLE_HDF5 "Enable native H5AD reading and writing" ON)
option(ANNDATA_ENABLE_ZARR "Enable native Zarr v2/v3 reading and writing" ON)
set(ANNDATA_DEPENDENCY_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/build/dependencies/usr" CACHE PATH
  "Optional private prefix containing storage development dependencies")
if(EXISTS "${ANNDATA_DEPENDENCY_PREFIX}")
  list(PREPEND CMAKE_PREFIX_PATH "${ANNDATA_DEPENDENCY_PREFIX}")
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(ANNDATA_ENABLE_HDF5 OR ANNDATA_ENABLE_ZARR)
  find_package(PkgConfig REQUIRED)
endif()
if(ANNDATA_ENABLE_HDF5)
  pkg_check_modules(ANNDATA_HDF5 REQUIRED IMPORTED_TARGET hdf5>=1.10)
endif()
if(ANNDATA_ENABLE_ZARR)
  pkg_check_modules(ANNDATA_ZLIB REQUIRED IMPORTED_TARGET zlib)
  pkg_check_modules(ANNDATA_ZSTD REQUIRED IMPORTED_TARGET libzstd)
  find_package(Blosc REQUIRED)
  find_package(nlohmann_json 3.11 CONFIG REQUIRED)
endif()

add_library(anndata)
add_library(anndata::anndata ALIAS anndata)
target_compile_features(anndata PUBLIC cxx_std_20)
target_compile_definitions(anndata PUBLIC
  ANNDATA_HAS_HDF5=$<BOOL:${ANNDATA_ENABLE_HDF5}>
  ANNDATA_HAS_ZARR=$<BOOL:${ANNDATA_ENABLE_ZARR}>)
set_target_properties(anndata PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME anndata_binary)
target_sources(anndata
  PUBLIC FILE_SET CXX_MODULES BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" FILES
    src/anndata.cppm
    src/anndata.core.cppm
    src/anndata.arrays.cppm
    src/anndata.strings.cppm
    src/anndata.data.cppm
    src/anndata.backed.cppm
  PRIVATE src/arrays.cpp src/strings.cpp src/data.cpp src/backed.cpp
    src/backed_array.cpp src/backed_codec.cpp src/backed_structure.cpp src/backed_writer.cpp
    src/backed_index.cpp src/backed_copy.cpp src/backed_category.cpp src/backed_colors.cpp)
target_sources(anndata PRIVATE FILE_SET backed_internal TYPE CXX_MODULES
  BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" FILES src/anndata.backed.storage.cppm)
if(ANNDATA_ENABLE_HDF5)
  target_sources(anndata
    PUBLIC FILE_SET CXX_MODULES FILES src/anndata.h5ad.cppm
    PRIVATE src/h5ad.cpp src/backed_hdf5.cpp)
  target_link_libraries(anndata PRIVATE PkgConfig::ANNDATA_HDF5)
endif()
if(ANNDATA_ENABLE_ZARR)
  target_sources(anndata PUBLIC FILE_SET CXX_MODULES FILES src/anndata.zarr.cppm
    PRIVATE src/zarr.cpp src/zarr_engine.cpp src/zarr_codecs.cpp src/zarr_sharding.cpp src/backed_zarr.cpp)
  target_link_libraries(anndata PRIVATE PkgConfig::ANNDATA_ZLIB PkgConfig::ANNDATA_ZSTD Blosc::Blosc
    "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>")
endif()
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
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindBlosc.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/anndata)
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/anndata)

if(ANNDATA_BUILD_EXAMPLES)
  add_executable(anndata_example examples/foundation.cpp)
  target_link_libraries(anndata_example PRIVATE anndata::anndata)
endif()
if(ANNDATA_BUILD_TESTS)
  enable_testing()
  foreach(_suite arrays data strings)
    add_executable(test_${_suite} tests/test_${_suite}.cpp)
    target_link_libraries(test_${_suite} PRIVATE anndata::anndata)
    add_test(NAME ${_suite} COMMAND test_${_suite})
  endforeach()
  add_executable(test_backed tests/test_backed.cpp)
  target_link_libraries(test_backed PRIVATE anndata::anndata)
  if(ANNDATA_ENABLE_HDF5)
    target_link_libraries(test_backed PRIVATE PkgConfig::ANNDATA_HDF5)
    target_compile_definitions(test_backed PRIVATE ANNDATA_TEST_HDF5=1)
  endif()
  add_test(NAME backed COMMAND test_backed "${CMAKE_CURRENT_BINARY_DIR}/backed-tests")
  add_executable(test_backed_categories tests/test_backed_categories.cpp)
  target_link_libraries(test_backed_categories PRIVATE anndata::anndata)
  add_test(NAME backed_categories COMMAND test_backed_categories "${CMAKE_CURRENT_BINARY_DIR}/backed-category-tests")
  if(ANNDATA_ENABLE_HDF5)
    add_executable(test_h5ad tests/test_h5ad.cpp)
    target_link_libraries(test_h5ad PRIVATE anndata::anndata)
    add_test(NAME h5ad COMMAND test_h5ad "${CMAKE_CURRENT_BINARY_DIR}/h5ad-tests")
    add_executable(test_backed_h5ad tests/test_backed_h5ad.cpp)
    target_link_libraries(test_backed_h5ad PRIVATE anndata::anndata)
    add_test(NAME backed_h5ad COMMAND test_backed_h5ad "${CMAKE_CURRENT_BINARY_DIR}/h5ad-backed-tests")
  endif()
  if(ANNDATA_BUILD_EXAMPLES)
    add_test(NAME import_only_example COMMAND anndata_example)
  endif()
  add_test(NAME installed_consumer COMMAND ${CMAKE_COMMAND}
    -DANNDATA_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DANNDATA_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DANNDATA_COMPILER=${CMAKE_CXX_COMPILER}
    -DANNDATA_CONFIG=$<CONFIG>
    -DANNDATA_HDF5_ENABLED=${ANNDATA_ENABLE_HDF5}
    -DANNDATA_ZARR_ENABLED=${ANNDATA_ENABLE_ZARR}
    -DANNDATA_DEPENDENCY_PREFIX=${ANNDATA_DEPENDENCY_PREFIX}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test-installed-consumer.cmake)
endif()
if(ANNDATA_ENABLE_INTEROP_TESTS)
  enable_testing()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_executable(interop_driver tests/interop_driver.cpp)
  target_link_libraries(interop_driver PRIVATE anndata::anndata)
  add_test(NAME python_semantics COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_semantics.py $<TARGET_FILE:interop_driver>)
  if(ANNDATA_ENABLE_HDF5)
    add_executable(h5ad_interop_driver tests/h5ad_interop_driver.cpp)
    target_link_libraries(h5ad_interop_driver PRIVATE anndata::anndata)
    add_test(NAME python_h5ad COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_h5ad.py $<TARGET_FILE:h5ad_interop_driver>)
  endif()
  if(ANNDATA_ENABLE_HDF5 AND ANNDATA_ENABLE_ZARR)
    add_executable(storage_interop_driver tests/storage_interop_driver.cpp)
    target_link_libraries(storage_interop_driver PRIVATE anndata::anndata)
    add_test(NAME python_storage COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_storage.py $<TARGET_FILE:storage_interop_driver>)
  endif()
endif()
if(ANNDATA_BUILD_BENCHMARKS)
  find_package(Threads REQUIRED)
  add_executable(anndata_selection_probe benchmarks/selection.cpp)
  target_link_libraries(anndata_selection_probe PRIVATE anndata::anndata)
  add_executable(memory_benchmark benchmarks/memory.cpp)
  target_link_libraries(memory_benchmark PRIVATE anndata::anndata Threads::Threads)
  if(ANNDATA_ENABLE_HDF5)
    add_executable(h5ad_benchmark benchmarks/h5ad.cpp)
    target_link_libraries(h5ad_benchmark PRIVATE anndata::anndata)
  endif()
  if(ANNDATA_ENABLE_HDF5 AND ANNDATA_ENABLE_ZARR)
    add_executable(storage_benchmark benchmarks/storage.cpp)
    target_link_libraries(storage_benchmark PRIVATE anndata::anndata Threads::Threads)
  endif()
endif()
