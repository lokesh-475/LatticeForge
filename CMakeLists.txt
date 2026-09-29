cmake_minimum_required(VERSION 3.10)
project(LatticeForge VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(lattice_forge src/main.cpp src/grid.cpp)
target_include_directories(lattice_forge PRIVATE include)
