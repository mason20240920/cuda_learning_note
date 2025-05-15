cmake_minimum_required(VERSION 3.29)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "80")
project(cuda_universal_demo VERSION 0.1.0 LANGUAGES CUDA CXX C)

find_package(CUDAToolkit)

add_subdirectory(1_cuda_sgemm)

set(CMAKE_CXX_STANDARD 17)

#add_executable(cuda_universal_demo main.cpp)
