cmake_minimum_required(VERSION 3.2)
project(pierank)

add_subdirectory(pagerank)
set(SOURCE_FILES main.cc)

add_executable(pierank ${SOURCE_FILES})
target_link_libraries(pierank pagerank)
install(TARGETS pierank DESTINATION ${PIERANK_INSTALL_BIN_DIR})