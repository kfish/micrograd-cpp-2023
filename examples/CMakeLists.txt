cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

enable_testing()

project(examples)

get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)

include_directories(
    ${PARENT_DIR}/include
    )

add_executable(regression0 regression0.cpp)
add_test(NAME regression0
         COMMAND regression0)

add_executable(binary-classifier binary-classifier.cpp)
add_test(NAME binary-classifier
         COMMAND binary-classifier)
