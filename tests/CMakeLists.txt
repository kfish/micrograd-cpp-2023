cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

enable_testing()

project(tests)

get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)

include_directories(
    ${PARENT_DIR}/include
    )

add_executable(value-test value-test.cpp)
add_test(NAME value-test
         COMMAND value-test)

add_executable(graph-test graph-test.cpp)
add_test(NAME graph-test
         COMMAND graph-test)

add_executable(neuron-test neuron-test.cpp)
add_test(NAME neuron-test
         COMMAND neuron-test)

add_executable(multivariate-test multivariate-test.cpp)
add_test(NAME multivariate-test
         COMMAND multivariate-test)
