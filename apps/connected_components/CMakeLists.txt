cmake_minimum_required(VERSION 3.2)
project(connected_components C CXX)

set(SOURCE_FILES connected_components.cc)

add_executable(connected_components ${SOURCE_FILES})
target_link_libraries(connected_components gflags ${glog_LDFLAGS} ${PIERANK_EXTERNAL_LIBRARIES})
add_dependencies(connected_components ${PIERANK_EXTERNAL_DEPENDENCIES})

install(TARGETS connected_components DESTINATION ${PIERANK_INSTALL_LIB_DIR})
