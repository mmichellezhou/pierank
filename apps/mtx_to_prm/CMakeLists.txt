cmake_minimum_required(VERSION 3.2)
project(mtx_to_prm C CXX)

set(SOURCE_FILES mtx_to_prm.cc)

add_executable(mtx_to_prm ${SOURCE_FILES})
target_link_libraries(mtx_to_prm gflags ${glog_LDFLAGS} ${PIERANK_EXTERNAL_LIBRARIES})
add_dependencies(mtx_to_prm ${PIERANK_EXTERNAL_DEPENDENCIES})

install(TARGETS mtx_to_prm DESTINATION ${PIERANK_INSTALL_LIB_DIR})
