cmake_minimum_required(VERSION 3.5)

project(MCTS)

include_directories(include)
include_directories(include/tree)
include_directories(src/manil)
add_compile_options(-pthread)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


file(GLOB_RECURSE SOURCES "src/*.cpp" main.cpp)
add_executable(example ${SOURCES})