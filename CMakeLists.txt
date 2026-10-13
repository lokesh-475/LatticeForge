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
    src/cuda/kawasaki_kernels.cu
    src/cuda/cuda_manager.cu
)
target_include_directories(lattice_forge PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(lattice_forge_cuda
    src/main_cuda.cpp
    src/grid.cpp 
    src/potts_cpu.cpp 
    src/kawasaki_cpu.cpp
    src/cuda/potts_kernels.cu
    src/cuda/kawasaki_kernels.cu
    src/cuda/cuda_manager.cu
)
target_include_directories(lattice_forge_cuda PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_custom_command(TARGET lattice_forge_cuda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:lattice_forge_cuda>
    ${CMAKE_SOURCE_DIR}/lattice_forge_cuda.exe
)

add_executable(benchmark_engine
    benchmarks/benchmark_engine.cu
    src/grid.cpp 
    src/potts_cpu.cpp 
    src/kawasaki_cpu.cpp
    src/cuda/potts_kernels.cu
    src/cuda/kawasaki_kernels.cu
    src/cuda/cuda_manager.cu
)
target_include_directories(benchmark_engine PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_custom_command(TARGET benchmark_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:benchmark_engine>
    ${CMAKE_SOURCE_DIR}/benchmark_engine.exe
)


enable_testing()
add_executable(test_parity 
    tests/test_parity.cpp 
    src/grid.cpp 
    src/potts_cpu.cpp 
    src/kawasaki_cpu.cpp
    src/cuda/potts_kernels.cu
    src/cuda/kawasaki_kernels.cu
    src/cuda/cuda_manager.cu
)
target_include_directories(test_parity PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_test(NAME ParityTest COMMAND test_parity)

