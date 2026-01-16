cmake_minimum_required(VERSION 3.18)

if(NOT TARGET rmsynth::core)
    message(FATAL_ERROR "rmsynth::core target not found. Build from project root.")
endif()

add_library(rmsynth_cxx INTERFACE)
target_include_directories(rmsynth_cxx
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cxx>
        $<INSTALL_INTERFACE:include/rmsynth>
)
target_link_libraries(rmsynth_cxx INTERFACE rmsynth::core)
add_library(rmsynth::cxx ALIAS rmsynth_cxx)

add_library(rmsynth_c STATIC
    c/rmsynth.cpp
)
target_include_directories(rmsynth_c
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/c>
        $<INSTALL_INTERFACE:include/rmsynth>
)
target_link_libraries(rmsynth_c PRIVATE rmsynth::core)
add_library(rmsynth::c ALIAS rmsynth_c)

find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(rmcore python/bindings.cpp)
    target_link_libraries(rmcore PRIVATE rmsynth::core)
    target_include_directories(rmcore PRIVATE ${CMAKE_SOURCE_DIR}/src/core)

    install(TARGETS rmcore
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
        ARCHIVE DESTINATION .
    )
    
    add_library(rmsynth::python ALIAS rmcore)
else()
    message(STATUS "pybind11 not found, Python bindings will not be built")
endif()
