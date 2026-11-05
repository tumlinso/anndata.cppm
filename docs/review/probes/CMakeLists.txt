cmake_minimum_required(VERSION 3.28)
project(record_validation_probe LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
find_package(anndatapp 0.1 CONFIG REQUIRED)

add_executable(record_validation record_validation.cpp)
target_link_libraries(record_validation PRIVATE anndatapp::anndatapp)
