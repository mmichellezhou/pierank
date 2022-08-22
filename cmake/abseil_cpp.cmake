include(ExternalProject)

set(abseil_cpp_INCLUDE_DIR ${CMAKE_BINARY_DIR}/abseil_cpp/src/abseil_cpp)
set(abseil_cpp_URL https://github.com/abseil/abseil-cpp/archive/refs/tags/20220623.0.tar.gz)
set(abseil_cpp_HASH SHA256=4208129b49006089ba1d6710845a45e31c59b0ab6bff9e5788a87f55c5abd602)
set(abseil_cpp_BUILD ${CMAKE_BINARY_DIR}/abseil_cpp/src/abseil_cpp)

set(abseil_cpp_STATIC_LIBRARIES
    ${abseil_cpp_BUILD}/absl/container/libabsl_hashtablez_sampler.a
    ${abseil_cpp_BUILD}/absl/container/libabsl_raw_hash_set.a

    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_commandlineflag_internal.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_commandlineflag.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_config.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_marshalling.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_parse.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_private_handle_accessor.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_program_name.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_reflection.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_usage.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_usage_internal.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags_internal.a
    ${abseil_cpp_BUILD}/absl/flags/libabsl_flags.a

    ${abseil_cpp_BUILD}/absl/hash/libabsl_city.a
    ${abseil_cpp_BUILD}/absl/hash/libabsl_hash.a
    ${abseil_cpp_BUILD}/absl/hash/libabsl_low_level_hash.a

    ${abseil_cpp_BUILD}/absl/random/libabsl_random_distributions.a
    ${abseil_cpp_BUILD}/absl/random/libabsl_random_internal_distribution_test_util.a
    ${abseil_cpp_BUILD}/absl/random/libabsl_random_internal_platform.a
    ${abseil_cpp_BUILD}/absl/random/libabsl_random_internal_pool_urbg.a
    ${abseil_cpp_BUILD}/absl/random/libabsl_random_internal_randen.a
    ${abseil_cpp_BUILD}/absl/random/libabsl_random_internal_randen_hwaes.a
    ${abseil_cpp_BUILD}/absl/random/libabsl_random_internal_randen_hwaes_impl.a
    ${abseil_cpp_BUILD}/absl/random/libabsl_random_internal_randen_slow.a
    ${abseil_cpp_BUILD}/absl/random/libabsl_random_internal_seed_material.a
    ${abseil_cpp_BUILD}/absl/random/libabsl_random_seed_gen_exception.a
    ${abseil_cpp_BUILD}/absl/random/libabsl_random_seed_sequences.a

    ${abseil_cpp_BUILD}/absl/status/libabsl_status.a
    ${abseil_cpp_BUILD}/absl/status/libabsl_statusor.a

    ${abseil_cpp_BUILD}/absl/strings/libabsl_cord.a
    ${abseil_cpp_BUILD}/absl/strings/libabsl_cord_internal.a
    ${abseil_cpp_BUILD}/absl/strings/libabsl_cordz_functions.a
    ${abseil_cpp_BUILD}/absl/strings/libabsl_cordz_info.a # before cordz_handle
    ${abseil_cpp_BUILD}/absl/strings/libabsl_cordz_handle.a
    ${abseil_cpp_BUILD}/absl/strings/libabsl_cordz_sample_token.a
    ${abseil_cpp_BUILD}/absl/strings/libabsl_str_format_internal.a
    ${abseil_cpp_BUILD}/absl/strings/libabsl_strings.a
    ${abseil_cpp_BUILD}/absl/strings/libabsl_strings_internal.a

    ${abseil_cpp_BUILD}/absl/synchronization/libabsl_graphcycles_internal.a
    ${abseil_cpp_BUILD}/absl/synchronization/libabsl_synchronization.a

    ${abseil_cpp_BUILD}/absl/time/libabsl_civil_time.a
    ${abseil_cpp_BUILD}/absl/time/libabsl_time.a
    ${abseil_cpp_BUILD}/absl/time/libabsl_time_zone.a

    ${abseil_cpp_BUILD}/absl/types/libabsl_bad_any_cast_impl.a
    ${abseil_cpp_BUILD}/absl/types/libabsl_bad_optional_access.a
    ${abseil_cpp_BUILD}/absl/types/libabsl_bad_variant_access.a

    # profiling, numeric, debugging, and base should come after other libs
    ${abseil_cpp_BUILD}/absl/profiling/libabsl_exponential_biased.a
    ${abseil_cpp_BUILD}/absl/profiling/libabsl_periodic_sampler.a

    ${abseil_cpp_BUILD}/absl/numeric/libabsl_int128.a

    ${abseil_cpp_BUILD}/absl/debugging/libabsl_examine_stack.a
    ${abseil_cpp_BUILD}/absl/debugging/libabsl_failure_signal_handler.a
    ${abseil_cpp_BUILD}/absl/debugging/libabsl_leak_check.a
    ${abseil_cpp_BUILD}/absl/debugging/libabsl_stacktrace.a
    ${abseil_cpp_BUILD}/absl/debugging/libabsl_symbolize.a
    ${abseil_cpp_BUILD}/absl/debugging/libabsl_debugging_internal.a
    ${abseil_cpp_BUILD}/absl/debugging/libabsl_demangle_internal.a

    ${abseil_cpp_BUILD}/absl/base/libabsl_base.a
    ${abseil_cpp_BUILD}/absl/base/libabsl_log_severity.a
    ${abseil_cpp_BUILD}/absl/base/libabsl_malloc_internal.a
    ${abseil_cpp_BUILD}/absl/base/libabsl_raw_logging_internal.a
    ${abseil_cpp_BUILD}/absl/base/libabsl_scoped_set_env.a
    ${abseil_cpp_BUILD}/absl/base/libabsl_spinlock_wait.a
    ${abseil_cpp_BUILD}/absl/base/libabsl_strerror.a
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
if (APPLE)
  list(APPEND PIERANK_EXTERNAL_LIBRARIES "-framework Foundation")
endif()
list(APPEND PIERANK_EXTERNAL_DEPENDENCIES abseil_cpp)

