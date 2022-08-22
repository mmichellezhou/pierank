cmake_minimum_required(VERSION 3.2)
project(pierank_tests)

add_subdirectory(lib/googletest)

include_directories(${PIERANK_HEADERS_DIR})
include_directories(lib/googletest/googletest/include)
include_directories(lib/googletest/googlemock/include)

set(SOURCE_FILES pierank_tests_main.cc
    src/pagerank_tests.cc
    src/matrix_market_io_tests.cc
    src/flex_index_tests.cc
    src/sparse_matrix_tests.cc
    src/thread_pool_tests.cc)

add_executable(pierank_tests ${SOURCE_FILES})
target_link_libraries(pierank_tests gtest glog::glog ${PIERANK_EXTERNAL_LIBRARIES})
add_dependencies(pierank_tests ${PIERANK_EXTERNAL_DEPENDENCIES})
install(TARGETS pierank_tests DESTINATION bin)
