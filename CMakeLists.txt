cmake_minimum_required(VERSION 3.2)

project(pierank)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(PIERANK_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PIERANK_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(PIERANK_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(PIERANK_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/pagerank)

include_directories(${PIERANK_INSTALL_INCLUDE_DIR})
include_directories(${PIERANK_HEADERS_DIR})

add_subdirectory(src)
add_subdirectory(test)


