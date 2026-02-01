cmake_minimum_required(VERSION 3.18)
project(rmsynth LANGUAGES C CXX)

option(RM_ENABLE_OPENMP "Enable OpenMP parallelism" OFF)
option(RM_ENABLE_TBB "Enable TBB parallelism" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)