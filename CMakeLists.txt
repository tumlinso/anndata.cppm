cmake_minimum_required(VERSION 3.28)
project(anndatapp VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18)
  message(FATAL_ERROR "The initial supported toolchain is Clang 18 or newer. Configure with -DCMAKE_CXX_COMPILER=clang++-18 and Ninja.")
endif()
if(NOT CMAKE_GENERATOR MATCHES "Ninja")
  message(FATAL_ERROR "Use Ninja (>=1.11) for C++20 module dependency scanning.")
endif()
string(REGEX MATCH "^[0-9]+" _anndatapp_clang_major "${CMAKE_CXX_COMPILER_VERSION}")
if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
  find_program(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS
    NAMES clang-scan-deps-${_anndatapp_clang_major} clang-scan-deps
    REQUIRED)
endif()
execute_process(COMMAND "${CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS}" --version
  RESULT_VARIABLE _scanner_result OUTPUT_VARIABLE _scanner_version ERROR_VARIABLE _scanner_error)
string(REGEX MATCH "[Vv]ersion[ ]+([0-9]+)" _scanner_match "${_scanner_version}")
if(NOT _scanner_result EQUAL 0 OR NOT _scanner_match OR NOT CMAKE_MATCH_1 STREQUAL _anndatapp_clang_major)
  message(FATAL_ERROR "clang-scan-deps must match Clang ${_anndatapp_clang_major}: ${_scanner_version}${_scanner_error}")
endif()

option(ANNDATAPP_BUILD_TESTS "Build standalone native tests" ${PROJECT_IS_TOP_LEVEL})
option(ANNDATAPP_BUILD_EXAMPLES "Build import-only examples" ${PROJECT_IS_TOP_LEVEL})
option(ANNDATAPP_BUILD_BENCHMARKS "Build numerical memory benchmarks" OFF)
option(ANNDATAPP_ENABLE_INTEROP_TESTS "Compare semantics using Python AnnData fixtures" OFF)
option(ANNDATAPP_ENABLE_RECORD_INTEROP_TESTS "Compare records using the isolated Scanpy reference" OFF)
option(ANNDATAPP_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ANNDATAPP_ENABLE_HDF5 "Enable native H5AD reading and writing" ON)
option(ANNDATAPP_ENABLE_ZARR "Enable native Zarr v2/v3 reading and writing" ON)
if(ANNDATAPP_ENABLE_RECORD_INTEROP_TESTS AND (NOT ANNDATAPP_ENABLE_INTEROP_TESTS OR NOT ANNDATAPP_ENABLE_HDF5))
  message(FATAL_ERROR "Record interoperability tests require ANNDATAPP_ENABLE_INTEROP_TESTS and ANNDATAPP_ENABLE_HDF5.")
endif()
set(ANNDATAPP_DEPENDENCY_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/build/dependencies/usr" CACHE PATH
  "Optional private prefix containing storage development dependencies")
if(EXISTS "${ANNDATAPP_DEPENDENCY_PREFIX}")
  list(PREPEND CMAKE_PREFIX_PATH "${ANNDATAPP_DEPENDENCY_PREFIX}")
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(ANNDATAPP_ENABLE_HDF5 OR ANNDATAPP_ENABLE_ZARR)
  find_package(PkgConfig REQUIRED)
endif()
if(ANNDATAPP_ENABLE_HDF5)
  pkg_check_modules(ANNDATAPP_HDF5 REQUIRED IMPORTED_TARGET hdf5>=1.10)
endif()
if(ANNDATAPP_ENABLE_ZARR)
  pkg_check_modules(ANNDATAPP_ZLIB REQUIRED IMPORTED_TARGET zlib)
  pkg_check_modules(ANNDATAPP_ZSTD REQUIRED IMPORTED_TARGET libzstd)
  find_package(Blosc REQUIRED)
  find_package(nlohmann_json 3.11 CONFIG REQUIRED)
endif()

add_library(anndatapp_core)
add_library(anndatapp::core ALIAS anndatapp_core)
target_compile_features(anndatapp_core PUBLIC cxx_std_20)
set_target_properties(anndatapp_core PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME core_binary)
target_sources(anndatapp_core
  PUBLIC FILE_SET CXX_MODULES BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" FILES
    src/anndatapp.core.cppm src/anndatapp.arrays.cppm src/anndatapp.strings.cppm
    src/anndatapp.records.cppm src/anndatapp.data.cppm
  PRIVATE src/arrays.cpp src/strings.cpp src/records.cpp src/data.cpp)

add_library(anndatapp)
add_library(anndatapp::anndatapp ALIAS anndatapp)
target_link_libraries(anndatapp PUBLIC anndatapp_core)
target_compile_features(anndatapp PUBLIC cxx_std_20)
target_compile_definitions(anndatapp PUBLIC
  ANNDATAPP_HAS_HDF5=$<BOOL:${ANNDATAPP_ENABLE_HDF5}>
  ANNDATAPP_HAS_ZARR=$<BOOL:${ANNDATAPP_ENABLE_ZARR}>)
set_target_properties(anndatapp PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME anndatapp_binary)
target_sources(anndatapp
  PUBLIC FILE_SET CXX_MODULES BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" FILES
    src/anndatapp.cppm src/anndatapp.backed.cppm src/anndatapp.streaming.cppm
  PRIVATE src/backed.cpp src/backed_array.cpp src/backed_codec.cpp src/backed_structure.cpp src/backed_writer.cpp
    src/backed_index.cpp src/backed_copy.cpp src/backed_category.cpp src/backed_colors.cpp
    src/backed_records.cpp src/backed_geometry.cpp src/streaming.cpp)
target_sources(anndatapp PRIVATE FILE_SET backed_internal TYPE CXX_MODULES
  BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" FILES src/anndatapp.backed.storage.cppm)
if(ANNDATAPP_ENABLE_HDF5)
  target_sources(anndatapp
    PUBLIC FILE_SET CXX_MODULES FILES src/anndatapp.h5ad.cppm
    PRIVATE src/h5ad.cpp src/backed_hdf5.cpp)
  target_link_libraries(anndatapp PRIVATE PkgConfig::ANNDATAPP_HDF5)
endif()
if(ANNDATAPP_ENABLE_ZARR)
  target_sources(anndatapp PUBLIC FILE_SET CXX_MODULES FILES src/anndatapp.zarr.cppm
    PRIVATE src/zarr.cpp src/zarr_engine.cpp src/zarr_codecs.cpp src/zarr_sharding.cpp src/backed_zarr.cpp)
  target_link_libraries(anndatapp PRIVATE PkgConfig::ANNDATAPP_ZLIB PkgConfig::ANNDATAPP_ZSTD Blosc::Blosc
    "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>")
endif()
target_compile_options(anndatapp_core PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(anndatapp PRIVATE -Wall -Wextra -Wpedantic)
if(ANNDATAPP_ENABLE_SANITIZERS)
  target_compile_options(anndatapp_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(anndatapp_core PUBLIC -fsanitize=address,undefined)
  target_compile_options(anndatapp PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(anndatapp PUBLIC -fsanitize=address,undefined)
endif()

install(TARGETS anndatapp_core EXPORT anndatappCoreTargets
  FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/anndatapp/modules
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT anndatappCoreTargets NAMESPACE anndatapp:: CXX_MODULES_DIRECTORY core-cxx-modules
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/anndatapp)
install(TARGETS anndatapp EXPORT anndatappTargets
  FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/anndatapp/modules
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT anndatappTargets NAMESPACE anndatapp:: CXX_MODULES_DIRECTORY cxx-modules
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/anndatapp)
configure_package_config_file(cmake/anndatappConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/anndatappConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/anndatapp)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/anndatappConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/anndatappConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/anndatappConfigVersion.cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindBlosc.cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/anndatappModuleProvider.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/anndatapp)
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/anndatapp)

if(ANNDATAPP_BUILD_EXAMPLES)
  add_executable(anndatapp_example examples/foundation.cpp)
  target_link_libraries(anndatapp_example PRIVATE anndatapp::anndatapp)
endif()
if(ANNDATAPP_BUILD_TESTS)
  enable_testing()
  find_package(Threads REQUIRED)
  foreach(_suite arrays data strings records copy_into)
    add_executable(test_${_suite} tests/test_${_suite}.cpp)
    target_link_libraries(test_${_suite} PRIVATE anndatapp::anndatapp)
    add_test(NAME ${_suite} COMMAND test_${_suite})
  endforeach()
  foreach(_suite geometry transfer_context streaming)
    add_executable(test_${_suite} tests/test_${_suite}.cpp)
    target_link_libraries(test_${_suite} PRIVATE anndatapp::anndatapp Threads::Threads)
    add_test(NAME ${_suite} COMMAND test_${_suite} "${CMAKE_CURRENT_BINARY_DIR}/${_suite}-tests")
  endforeach()
  add_executable(test_backed tests/test_backed.cpp)
  target_link_libraries(test_backed PRIVATE anndatapp::anndatapp)
  if(ANNDATAPP_ENABLE_HDF5)
    target_link_libraries(test_backed PRIVATE PkgConfig::ANNDATAPP_HDF5)
    target_compile_definitions(test_backed PRIVATE ANNDATAPP_TEST_HDF5=1)
  endif()
  add_test(NAME backed COMMAND test_backed "${CMAKE_CURRENT_BINARY_DIR}/backed-tests")
  add_executable(test_backed_categories tests/test_backed_categories.cpp)
  target_link_libraries(test_backed_categories PRIVATE anndatapp::anndatapp)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(test_backed_categories PRIVATE -Wl,--export-dynamic-symbol=pread)
  endif()
  add_test(NAME backed_categories COMMAND test_backed_categories "${CMAKE_CURRENT_BINARY_DIR}/backed-category-tests")
  if(ANNDATAPP_ENABLE_ZARR)
    add_executable(test_gather_transfer tests/test_gather_transfer.cpp)
    target_include_directories(test_gather_transfer PRIVATE src)
    target_link_libraries(test_gather_transfer PRIVATE anndatapp::anndatapp nlohmann_json::nlohmann_json)
    add_test(NAME gather_transfer COMMAND test_gather_transfer "${CMAKE_CURRENT_BINARY_DIR}/gather-transfer-tests")
    add_executable(test_shard_selection tests/test_shard_selection.cpp)
    target_include_directories(test_shard_selection PRIVATE src)
    target_link_libraries(test_shard_selection PRIVATE anndatapp::anndatapp nlohmann_json::nlohmann_json)
    add_test(NAME shard_selection COMMAND test_shard_selection "${CMAKE_CURRENT_BINARY_DIR}/shard-selection-tests")
    add_executable(test_zarr_sharding tests/test_zarr_sharding.cpp)
    target_include_directories(test_zarr_sharding PRIVATE src)
    target_link_libraries(test_zarr_sharding PRIVATE anndatapp::anndatapp nlohmann_json::nlohmann_json)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_link_options(test_zarr_sharding PRIVATE
        -Wl,--export-dynamic-symbol=pread -Wl,--export-dynamic-symbol=pwrite)
    endif()
    add_test(NAME zarr_sharding COMMAND test_zarr_sharding "${CMAKE_CURRENT_BINARY_DIR}/shard-tests")
    add_executable(test_record_zarr_rejection tests/test_record_zarr_rejection.cpp)
    target_link_libraries(test_record_zarr_rejection PRIVATE anndatapp::anndatapp)
    add_test(NAME record_zarr_rejection COMMAND test_record_zarr_rejection "${CMAKE_CURRENT_BINARY_DIR}/record-zarr-rejection")
  endif()
  if(ANNDATAPP_ENABLE_HDF5)
    add_executable(test_records_h5ad tests/test_records_h5ad.cpp)
    target_include_directories(test_records_h5ad PRIVATE src)
    target_link_libraries(test_records_h5ad PRIVATE anndatapp::anndatapp PkgConfig::ANNDATAPP_HDF5 ${CMAKE_DL_LIBS})
    add_test(NAME records_h5ad COMMAND test_records_h5ad "${CMAKE_CURRENT_BINARY_DIR}/records-h5ad-tests")
    if(APPLE)
      add_library(test_records_h5ad_observer SHARED tests/h5ad_read_observer.cpp)
      target_link_libraries(test_records_h5ad_observer PRIVATE PkgConfig::ANNDATAPP_HDF5)
      add_dependencies(test_records_h5ad test_records_h5ad_observer)
      set_tests_properties(records_h5ad PROPERTIES
        ENVIRONMENT "DYLD_INSERT_LIBRARIES=$<TARGET_FILE:test_records_h5ad_observer>")
    endif()
    add_executable(test_backed_records tests/test_backed_records.cpp)
    target_link_libraries(test_backed_records PRIVATE anndatapp::anndatapp)
    add_test(NAME backed_records COMMAND test_backed_records "${CMAKE_CURRENT_BINARY_DIR}/backed-record-tests")
    add_executable(test_h5ad tests/test_h5ad.cpp)
    target_link_libraries(test_h5ad PRIVATE anndatapp::anndatapp)
    add_test(NAME h5ad COMMAND test_h5ad "${CMAKE_CURRENT_BINARY_DIR}/h5ad-tests")
    add_executable(test_backed_h5ad tests/test_backed_h5ad.cpp)
    target_link_libraries(test_backed_h5ad PRIVATE anndatapp::anndatapp)
    add_test(NAME backed_h5ad COMMAND test_backed_h5ad "${CMAKE_CURRENT_BINARY_DIR}/h5ad-backed-tests")
  endif()
  if(ANNDATAPP_BUILD_EXAMPLES)
    add_test(NAME import_only_example COMMAND anndatapp_example)
  endif()
  add_test(NAME installed_consumer COMMAND ${CMAKE_COMMAND}
    -DANNDATAPP_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DANNDATAPP_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DANNDATAPP_COMPILER=${CMAKE_CXX_COMPILER}
    -DANNDATAPP_CONFIG=$<CONFIG>
    -DANNDATAPP_HDF5_ENABLED=${ANNDATAPP_ENABLE_HDF5}
    -DANNDATAPP_ZARR_ENABLED=${ANNDATAPP_ENABLE_ZARR}
    -DANNDATAPP_DEPENDENCY_PREFIX=${ANNDATAPP_DEPENDENCY_PREFIX}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test-installed-consumer.cmake)
endif()
if(ANNDATAPP_ENABLE_INTEROP_TESTS)
  enable_testing()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_executable(interop_driver tests/interop_driver.cpp)
  target_link_libraries(interop_driver PRIVATE anndatapp::anndatapp)
  add_test(NAME python_semantics COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_semantics.py $<TARGET_FILE:interop_driver>)
  if(ANNDATAPP_ENABLE_HDF5)
    if(ANNDATAPP_ENABLE_RECORD_INTEROP_TESTS)
      add_executable(records_interop_driver tests/records_interop_driver.cpp)
      target_link_libraries(records_interop_driver PRIVATE anndatapp::anndatapp)
      add_test(NAME python_records COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_records.py $<TARGET_FILE:records_interop_driver>)
    endif()
    add_executable(h5ad_interop_driver tests/h5ad_interop_driver.cpp)
    target_link_libraries(h5ad_interop_driver PRIVATE anndatapp::anndatapp)
    add_test(NAME python_h5ad COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_h5ad.py $<TARGET_FILE:h5ad_interop_driver>)
  endif()
  if(ANNDATAPP_ENABLE_HDF5 AND ANNDATAPP_ENABLE_ZARR)
    find_package(Threads REQUIRED)
    add_executable(streaming_interop_driver tests/streaming_interop_driver.cpp)
    target_link_libraries(streaming_interop_driver PRIVATE anndatapp::anndatapp Threads::Threads)
    add_test(NAME python_streaming COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_streaming.py $<TARGET_FILE:streaming_interop_driver>
      "${CMAKE_CURRENT_BINARY_DIR}/streaming-interop-artifacts")
    add_test(NAME python_streaming_threaded COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_streaming.py $<TARGET_FILE:streaming_interop_driver>
      "${CMAKE_CURRENT_BINARY_DIR}/streaming-threaded-artifacts" --threaded)
    add_executable(completeness_interop_driver tests/completeness_interop_driver.cpp)
    target_link_libraries(completeness_interop_driver PRIVATE anndatapp::anndatapp Threads::Threads)
    add_test(NAME python_completeness COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_completeness.py $<TARGET_FILE:completeness_interop_driver>
      "${CMAKE_CURRENT_BINARY_DIR}/completeness-interop-artifacts")
    add_test(NAME python_completeness_threaded COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_completeness.py $<TARGET_FILE:completeness_interop_driver>
      "${CMAKE_CURRENT_BINARY_DIR}/completeness-threaded-artifacts" --threaded)
    add_executable(storage_interop_driver tests/storage_interop_driver.cpp)
    target_link_libraries(storage_interop_driver PRIVATE anndatapp::anndatapp)
    add_test(NAME python_storage COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_storage.py $<TARGET_FILE:storage_interop_driver>)
  endif()
  if(ANNDATAPP_ENABLE_HDF5 OR ANNDATAPP_ENABLE_ZARR)
    add_executable(dtype_rejection_driver tests/dtype_rejection_driver.cpp)
    target_link_libraries(dtype_rejection_driver PRIVATE anndatapp::anndatapp)
    if(ANNDATAPP_ENABLE_HDF5)
      add_test(NAME python_unsupported_dtype_h5ad COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_unsupported_dtype.py
        $<TARGET_FILE:dtype_rejection_driver> h5ad
        "${CMAKE_CURRENT_BINARY_DIR}/unsupported-dtype-h5ad")
    endif()
    if(ANNDATAPP_ENABLE_ZARR)
      foreach(_format zarr2 zarr3)
        add_test(NAME python_unsupported_dtype_${_format} COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_unsupported_dtype.py
          $<TARGET_FILE:dtype_rejection_driver> ${_format}
          "${CMAKE_CURRENT_BINARY_DIR}/unsupported-dtype-${_format}")
      endforeach()
    endif()
  endif()
endif()
if(ANNDATAPP_BUILD_BENCHMARKS)
  find_package(Threads REQUIRED)
  if(ANNDATAPP_ENABLE_ZARR)
    add_executable(completeness_probe benchmarks/completeness.cpp)
    target_link_libraries(completeness_probe PRIVATE anndatapp::anndatapp ${CMAKE_DL_LIBS})
    add_executable(completeness_adaptive benchmarks/completeness_adaptive.cpp)
    target_link_libraries(completeness_adaptive PRIVATE anndatapp::anndatapp Threads::Threads ${CMAKE_DL_LIBS})
  endif()
  add_executable(streaming_benchmark benchmarks/streaming.cpp)
  target_compile_definitions(streaming_benchmark PRIVATE ANNDATAPP_STREAM_CANDIDATE=1)
  target_link_libraries(streaming_benchmark PRIVATE anndatapp::anndatapp Threads::Threads)
  add_executable(records_benchmark benchmarks/records.cpp)
  target_link_libraries(records_benchmark PRIVATE anndatapp::anndatapp)
  add_executable(anndatapp_selection_probe benchmarks/selection.cpp)
  target_link_libraries(anndatapp_selection_probe PRIVATE anndatapp::anndatapp)
  add_executable(memory_benchmark benchmarks/memory.cpp)
  target_link_libraries(memory_benchmark PRIVATE anndatapp::anndatapp Threads::Threads)
  if(ANNDATAPP_ENABLE_HDF5)
    add_executable(record_io_probe benchmarks/record_io_probe/driver.cpp)
    target_include_directories(record_io_probe PRIVATE benchmarks/record_io_probe)
    target_link_libraries(record_io_probe PRIVATE anndatapp::anndatapp ${CMAKE_DL_LIBS})
    add_executable(h5ad_benchmark benchmarks/h5ad.cpp)
    target_link_libraries(h5ad_benchmark PRIVATE anndatapp::anndatapp)
  endif()
  if(ANNDATAPP_ENABLE_HDF5 AND ANNDATAPP_ENABLE_ZARR)
    add_executable(storage_benchmark benchmarks/storage.cpp)
    target_link_libraries(storage_benchmark PRIVATE anndatapp::anndatapp Threads::Threads)
  endif()
endif()
