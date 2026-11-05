find_package(anndatapp 0.1 CONFIG REQUIRED COMPONENTS core)
add_executable(core_consumer ../../main.cpp)
target_link_libraries(core_consumer PRIVATE anndatapp::core)
