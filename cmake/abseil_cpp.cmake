include(ExternalProject)

set(abseil_cpp_INCLUDE_DIR ${CMAKE_BINARY_DIR}/abseil_cpp/src/abseil_cpp)
set(abseil_cpp_URL https://github.com/abseil/abseil-cpp/archive/refs/tags/20220623.0.tar.gz)
set(abseil_cpp_HASH SHA256=4208129b49006089ba1d6710845a45e31c59b0ab6bff9e5788a87f55c5abd602)
set(abseil_cpp_BUILD ${CMAKE_BINARY_DIR}/abseil_cpp/src/abseil_cpp)

set(abseil_cpp_STATIC_LIBRARIES
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_parse.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_usage_internal.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_usage.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_marshalling.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_config.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_internal.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_program_name.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags.a

    ${abseil_cpp_BUILD}/absl/strings/libabsl_strings.a
    ${abseil_cpp_BUILD}/absl/strings/libabsl_strings_internal.a
    ${abseil_cpp_BUILD}/absl/strings/libabsl_str_format_internal.a

    ${abseil_cpp_BUILD}/absl/synchronization/libabsl_synchronization.a

    ${abseil_cpp_BUILD}/absl/debugging/libabsl_symbolize.a
    ${abseil_cpp_BUILD}/absl/debugging/libabsl_failure_signal_handler.a
    ${abseil_cpp_BUILD}/absl/debugging/libabsl_examine_stack.a
    ${abseil_cpp_BUILD}/absl/debugging/libabsl_leak_check.a
    ${abseil_cpp_BUILD}/absl/debugging/libabsl_stacktrace.a
    ${abseil_cpp_BUILD}/absl/debugging/libabsl_demangle_internal.a
    ${abseil_cpp_BUILD}/absl/debugging/libabsl_debugging_internal.a

    ${abseil_cpp_BUILD}/absl/time/libabsl_time.a
    ${abseil_cpp_BUILD}/absl/time/libabsl_time_zone.a

    ${abseil_cpp_BUILD}/absl/numeric/libabsl_int128.a

    ${abseil_cpp_BUILD}/absl/base/libabsl_base.a
    ${abseil_cpp_BUILD}/absl/base/libabsl_malloc_internal.a
    ${abseil_cpp_BUILD}/absl/base/libabsl_raw_logging_internal.a
    ${abseil_cpp_BUILD}/absl/base/libabsl_spinlock_wait.a
    ${abseil_cpp_BUILD}/absl/base/libabsl_throw_delegate.a
    )

ExternalProject_Add(abseil_cpp
    PREFIX abseil_cpp
    URL ${abseil_cpp_URL}
    URL_HASH ${abseil_cpp_HASH}
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${abseil_cpp_STATIC_LIBRARIES}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release
    COMMAND ${CMAKE_COMMAND} --build . --config Release
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
    -DCMAKE_CXX_STANDARD:STRING=17
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=Release
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF)

include_directories(${abseil_cpp_INCLUDE_DIR})
message(STATUS ${abseil_cpp_INCLUDE_DIR})

list(APPEND PIERANK_EXTERNAL_LIBRARIES ${abseil_cpp_STATIC_LIBRARIES})

list(APPEND PIERANK_EXTERNAL_DEPENDENCIES abseil_cpp)

