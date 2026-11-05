find_package(anndatapp 1.0 CONFIG REQUIRED COMPONENTS core)
add_executable(core_records ../../records.cpp)
target_link_libraries(core_records PRIVATE anndatapp::core)
