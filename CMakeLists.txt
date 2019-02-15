cmake_minimum_required(VERSION 2.6)


PROJECT("LibSimd")

#libraries
add_subdirectory(libsimd/src/)


#tests
add_subdirectory(libsimd/test)


#Setup CMake to run tests
enable_testing()

add_test(NAME simdtest
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testdir 
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/output/bin/simdtest )
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testdir)
