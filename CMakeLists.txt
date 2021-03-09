cmake_minimum_required(VERSION 3.17)
project(Project)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_CXX_STANDARD 20)

add_executable(Project main.cpp side_functions.h block_centers.h)

target_link_libraries(Project ${CONAN_LIBS})