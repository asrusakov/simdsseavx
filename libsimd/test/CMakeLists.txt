cmake_minimum_required(VERSION 2.6)

include (../../buildutils/CMakeExecutableTemplate.txt)

set (LIBS  libsimd)

add_executable(simdtest simd_test.cpp) 

target_link_libraries(simdtest ${START_GROUP}  ${LIBS}  ${END_GROUP}  ${EXTERNAL_LIBS}) 

