cmake_minimum_required(VERSION 3.2)
project(pagerank_tests)

add_subdirectory(lib/googletest)

include_directories(${PIERANK_HEADERS_DIR})
include_directories(lib/googletest/googletest/include)

set(SOURCE_FILES main.cc
    src/pagerank_tests.cc
    src/matrix_market_io_tests.cc
    src/flex_index_tests.cc
    src/sparse_matrix_tests.cc)

add_executable(pagerank_tests ${SOURCE_FILES})
target_link_libraries(pagerank_tests pagerank gtest glog::glog)
install(TARGETS pagerank_tests DESTINATION bin)
