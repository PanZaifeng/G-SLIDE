cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(G_SLIDE LANGUAGES CXX CUDA)

include(FetchContent)

FetchContent_Declare(
  jsoncpp_static
  GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
  GIT_TAG 1.9.5)

FetchContent_MakeAvailable(jsoncpp_static)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRCS)

add_executable(runme ${SRCS})

target_include_directories(runme PRIVATE include)

set_target_properties(runme PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(runme PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                       ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(runme PUBLIC cxx_std_14)

target_link_libraries(runme PRIVATE jsoncpp_static)
target_link_libraries(runme PRIVATE cublas)
