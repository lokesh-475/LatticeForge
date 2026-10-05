cmake_minimum_required(VERSION 3.18)
project(LatticeForge VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

add_executable(lattice_forge 
    src/main.cpp 
    src/grid.cpp 
    src/potts_cpu.cpp 
    src/kawasaki_cpu.cpp
    src/cuda/potts_kernels.cu
    src/cuda/cuda_manager.cu
)
target_include_directories(lattice_forge PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

enable_testing()
add_executable(test_parity 
    tests/test_parity.cpp 
    src/grid.cpp 
    src/potts_cpu.cpp 
    src/kawasaki_cpu.cpp
    src/cuda/potts_kernels.cu
    src/cuda/cuda_manager.cu
)
target_include_directories(test_parity PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_test(NAME ParityTest COMMAND test_parity)
