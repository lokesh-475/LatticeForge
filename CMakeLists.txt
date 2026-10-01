cmake_minimum_required(VERSION 3.18)
project(LatticeForge VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 native)

add_executable(lattice_forge 
    src/main.cpp 
    src/grid.cpp 
    src/potts_cpu.cpp 
    src/kawasaki_cpu.cpp
)
target_include_directories(lattice_forge PRIVATE include)
