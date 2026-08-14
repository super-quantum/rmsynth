cmake_minimum_required(VERSION 3.20)

project(rmsynth_reference VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

option(RMSYNTH_BUILD_PYTHON "Build the Python extension" ON)
option(RMSYNTH_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

function(rmsynth_enable_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
    if(RMSYNTH_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(
      ${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
    )
    if(RMSYNTH_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

add_library(rmsynth_core STATIC cpp/src/core.cpp)
add_library(rmsynth_reference::core ALIAS rmsynth_core)
target_include_directories(rmsynth_core PUBLIC cpp/include)
target_compile_features(rmsynth_core PUBLIC cxx_std_20)
set_target_properties(rmsynth_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
rmsynth_enable_warnings(rmsynth_core)

if(RMSYNTH_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native MODULE cpp/src/bindings.cpp)
  target_link_libraries(_native PRIVATE rmsynth_reference::core)
  rmsynth_enable_warnings(_native)
  install(
    TARGETS _native
    LIBRARY DESTINATION rmsynth_reference
    RUNTIME DESTINATION rmsynth_reference
  )
endif()

if(BUILD_TESTING)
  add_executable(rmsynth_core_tests cpp/tests/core_tests.cpp)
  target_link_libraries(rmsynth_core_tests PRIVATE rmsynth_reference::core)
  rmsynth_enable_warnings(rmsynth_core_tests)
  add_test(NAME rmsynth_core_tests COMMAND rmsynth_core_tests)
endif()
