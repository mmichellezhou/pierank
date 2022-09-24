cmake_minimum_required(VERSION 3.2)
project(prm_print C CXX)

set(SOURCE_FILES prm_print.cc)

add_executable(prm_print ${SOURCE_FILES})
target_link_libraries(prm_print gflags ${glog_LDFLAGS} ${PIERANK_EXTERNAL_LIBRARIES})
add_dependencies(prm_print ${PIERANK_EXTERNAL_DEPENDENCIES})

install(TARGETS prm_print DESTINATION ${PIERANK_INSTALL_LIB_DIR})
