cmake_minimum_required(VERSION 3.2)

project(pierank)

OPTION(PIERANK_CONST_MEMCPY "Use constant-size memcpy" OFF)
IF(PIERANK_CONST_MEMCPY)
  MESSAGE("Use constant-size memcpy as needed")
  ADD_COMPILE_DEFINITIONS(PIERANK_USE_CONST_MEMCPY)
ENDIF(PIERANK_CONST_MEMCPY)

# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Threads REQUIRED)

find_package(gflags REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(glog REQUIRED libglog)

include_directories(${gflags_INCLUDE_DIR} ${glog_INCLUDE_DIRS})

include(abseil_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(PIERANK_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PIERANK_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(PIERANK_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(PIERANK_HEADERS_DIR ${PROJECT_SOURCE_DIR})

include_directories(${PIERANK_INSTALL_INCLUDE_DIR})
include_directories(${PIERANK_HEADERS_DIR})

add_subdirectory(pierank)
add_subdirectory(test)
add_subdirectory(apps)


