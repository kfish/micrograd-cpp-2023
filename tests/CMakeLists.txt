cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)
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

add_executable(multivariate-test multivariate-test.cpp)
add_test(NAME multivariate-test
         COMMAND multivariate-test)

add_executable(mac-test mac-test.cpp)
add_test(NAME mac-test
         COMMAND mac-test)
