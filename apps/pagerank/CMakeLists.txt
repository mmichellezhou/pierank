cmake_minimum_required(VERSION 3.2)
project(pagerank C CXX)

set(SOURCE_FILES pagerank.cc)

add_executable(pagerank ${SOURCE_FILES})
target_link_libraries(pagerank gflags glog::glog ${PIERANK_EXTERNAL_LIBRARIES})
add_dependencies(pagerank ${PIERANK_EXTERNAL_DEPENDENCIES})

install(TARGETS pagerank DESTINATION ${PIERANK_INSTALL_LIB_DIR})
