cmake_minimum_required(VERSION 3.2)
project(pagerank C CXX)

set(SOURCE_FILES
        pagerank.h
        pagerank.cc
        )

add_library(pagerank SHARED STATIC ${SOURCE_FILES})

install(TARGETS pagerank DESTINATION ${PIERANK_INSTALL_LIB_DIR})
install(FILES pagerank.h DESTINATION ${PIERANK_INSTALL_INCLUDE_DIR})
