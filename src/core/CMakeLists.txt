cmake_minimum_required(VERSION 3.18)
project(rmsynth_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(RMSYNTH_CORE_SOURCES
    anf.cpp
    decode_bruteforce.cpp
    dumer.cpp
    dumer_list.cpp
    dumer_list_chase.cpp
    rm_code.cpp
    rpa.cpp
    tdepth.cpp
)
set(RMSYNTH_CORE_HEADERS
    anf.h
    bitops.h
    decode_bruteforce.h
    dumer.h
    dumer_list.h
    dumer_list_chase.h
    parallel.h
    puncture.h
    rm_code.h
    rpa.h
    tdepth.h
)

add_library(rmsynth_core STATIC ${RMSYNTH_CORE_SOURCES})
target_include_directories(rmsynth_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/rmsynth/core>
)

set_target_properties(rmsynth_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(RM_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(rmsynth_core PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(rmsynth_core PUBLIC RM_HAVE_OPENMP=1)
    else()
        message(WARNING "RM_ENABLE_OPENMP=ON, but OpenMP not found, building without OpenMP")
    endif()
endif()

if(RM_ENABLE_TBB)
    find_package(TBB)
    if(TBB_FOUND)
        target_link_libraries(rmsynth_core PUBLIC TBB::tbb)
        target_compile_definitions(rmsynth_core PUBLIC RM_HAVE_TBB=1)
    else()
        message(WARNING "RM_ENABLE_TBB=ON, but TBB not found, building without TBB")
    endif()
endif()

add_library(rmsynth::core ALIAS rmsynth_core)
