cmake_minimum_required(VERSION 3.20)
project(ll_combine_bench LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
get_filename_component(DEEPEP_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
find_package(MPI REQUIRED COMPONENTS CXX)
include("${DEEPEP_SOURCE_DIR}/cmake/Dependencies.cmake")
find_file(SHMEM_MPI_PLUGIN aclshmem_bootstrap_mpi.so PATHS "${SHMEM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
option(DEEPEP_LL_COMBINE_PROFILE "Build optional Combine diagnostics" OFF)
option(DEEPEP_LL_OPS_BASE "Build the ops-transformer SHMEM adaptation" OFF)
option(DEEPEP_LL_COMBINE_ORIGINAL "Use the unaltered upstream Combine kernel" ON)
option(DEEPEP_LL_COMBINE_SMALL_BATCH "Cache partial tasks and trim idle lanes" OFF)
option(DEEPEP_LL_COMBINE_EARLY_REDUCE "Reduce ready contributions for BS <= 72" OFF)
option(DEEPEP_LL_COMBINE_REMOTE_FIRST "Return cross-host results before local-host results" OFF)
option(DEEPEP_LL_COMBINE_DEFER_RETURN "Overlap return staging with next-packet preparation" OFF)
set(DEEPEP_LL_COMBINE_RECV_CORES "0" CACHE STRING "0=half; explicit receive cores in [8,36]")
if(NOT DEEPEP_LL_COMBINE_RECV_CORES MATCHES "^(0|[89]|[12][0-9]|3[0-6])$")
    message(FATAL_ERROR "Receive cores must be 0 or an integer in [8,36]")
endif()
set(DEEPEP_LL_METADATA_CORES 4)
set(DEEPEP_LL_REFERENCE_PROFILE OFF)
set(DEEPEP_LL_STAGE_TIMER OFF)
set(DEEPEP_LL_COUNT_HISTOGRAM OFF)
if(DEEPEP_LL_OPS_BASE)
    if(DEEPEP_LL_COMBINE_ORIGINAL OR DEEPEP_LL_COMBINE_PROFILE OR DEEPEP_LL_COMBINE_SMALL_BATCH OR
       DEEPEP_LL_COMBINE_EARLY_REDUCE OR DEEPEP_LL_COMBINE_REMOTE_FIRST OR DEEPEP_LL_COMBINE_DEFER_RETURN OR
       NOT DEEPEP_LL_COMBINE_RECV_CORES STREQUAL "0")
        message(FATAL_ERROR "ops-shmem-base has no DeepEP experiment flags or instrumentation")
    endif()
    include("${CMAKE_CURRENT_LIST_DIR}/ops_shmem/source.cmake")
elseif(DEEPEP_LL_COMBINE_ORIGINAL)
    if(DEEPEP_LL_COMBINE_SMALL_BATCH OR DEEPEP_LL_COMBINE_EARLY_REDUCE OR DEEPEP_LL_COMBINE_PROFILE OR
       DEEPEP_LL_COMBINE_REMOTE_FIRST OR DEEPEP_LL_COMBINE_DEFER_RETURN OR
       NOT DEEPEP_LL_COMBINE_RECV_CORES STREQUAL "0")
        message(FATAL_ERROR "Original Combine cannot enable experimental options or profiling; use --variant opt")
    endif()
    include("${CMAKE_CURRENT_LIST_DIR}/baseline.cmake")
else()
    set(DEEPEP_LL_COMBINE_SOURCE "${DEEPEP_SOURCE_DIR}/csrc/kernels/combine/low_latency.cpp")
endif()
set(DEEPEP_LL_BENCH_GATE ON)
string(SHA256 _variant_tag "${CMAKE_BINARY_DIR}")
string(SUBSTRING "${_variant_tag}" 0 12 _variant_tag)
set(DEEPEP_DEVICE_OUTPUT_NAME "deepep_bench_${_variant_tag}")
set(DEEPEP_DEVICE_LOCAL_SYMBOLS ON)
include("${DEEPEP_SOURCE_DIR}/cmake/Device.cmake")
find_path(DEEPEP_PLATFORM_INCLUDE tiling/platform/platform_ascendc.h
    HINTS "${ASCEND_HOME_PATH}/asc/include" "${ASCEND_HOME_PATH}/include/op_common"
          "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include" NO_DEFAULT_PATH REQUIRED)
if(DEEPEP_LL_OPS_BASE)
    set(_bench_main ops_shmem/main.cpp)
else()
    set(_bench_main main.cpp)
endif()
add_library(ll_bench_variant MODULE ${_bench_main})
set_target_properties(ll_bench_variant PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
    NO_SONAME TRUE)
target_link_options(ll_bench_variant PRIVATE -Wl,-Bsymbolic)
if(DEEPEP_LL_OPS_BASE)
    target_compile_definitions(ll_bench_variant PRIVATE LL_OPS_REV="${LL_OPS_REV}")
endif()
if(DEEPEP_LL_COMBINE_ORIGINAL)
    target_sources(ll_bench_variant PRIVATE baseline_queries.cpp)
endif()
target_include_directories(ll_bench_variant PRIVATE "${DEEPEP_SOURCE_DIR}/csrc"
    "${DEEPEP_PLATFORM_INCLUDE}" "${ASCEND_HOME_PATH}/include" "${SHMEM_ROOT}/include")
target_link_directories(ll_bench_variant PRIVATE "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
# tiling_api is static in CANN 9.2 and uses platform symbols. Keep its provider
# afterwards so --as-needed retains libplatform when linking the executable.
target_link_libraries(ll_bench_variant PRIVATE MPI::MPI_CXX ascendcl shmem deepep_device
    tiling_api platform unified_dlog ${CMAKE_DL_LIBS} pthread)
target_compile_definitions(ll_bench_variant PRIVATE DEEPEP_LL_RDMA=1 DEEPEP_LL_METADATA_CORES=4
    DEEPEP_LL_COMBINE_ORIGINAL=$<BOOL:${DEEPEP_LL_COMBINE_ORIGINAL}>
    DEEPEP_LL_COMBINE_RECV_CORES=${DEEPEP_LL_COMBINE_RECV_CORES}
    $<$<BOOL:${DEEPEP_LL_COMBINE_SMALL_BATCH}>:DEEPEP_LL_COMBINE_SMALL_BATCH=1>
    $<$<BOOL:${DEEPEP_LL_COMBINE_EARLY_REDUCE}>:DEEPEP_LL_COMBINE_EARLY_REDUCE=1>)
target_compile_definitions(ll_bench_variant PRIVATE
    DEEPEP_BENCH_REMOTE_FIRST=$<BOOL:${DEEPEP_LL_COMBINE_REMOTE_FIRST}>
    DEEPEP_BENCH_DEFER_RETURN=$<BOOL:${DEEPEP_LL_COMBINE_DEFER_RETURN}>)
# Hash the compiled source tree, including uncommitted edits, for cross-rank checks.
file(GLOB_RECURSE _identity_files CONFIGURE_DEPENDS
    "${DEEPEP_SOURCE_DIR}/csrc/*.h" "${DEEPEP_SOURCE_DIR}/csrc/*.hpp" "${DEEPEP_SOURCE_DIR}/csrc/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/*.hpp" "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
list(SORT _identity_files)
set(_source_hashes "${LL_OPS_REV}")
foreach(_file IN LISTS _identity_files)
    file(SHA256 "${_file}" _hash)
    string(APPEND _source_hashes "${_hash}")
endforeach()
string(SHA256 _source_id "${_source_hashes}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_identity_files})
target_compile_definitions(ll_bench_variant PRIVATE DEEPEP_BENCH_SOURCE_ID="${_source_id}")
add_executable(ll_combine_bench runner.cpp "${DEEPEP_SOURCE_DIR}/csrc/runtime/session.cpp")
target_include_directories(ll_combine_bench PRIVATE "${DEEPEP_SOURCE_DIR}/csrc"
    "${DEEPEP_PLATFORM_INCLUDE}" "${ASCEND_HOME_PATH}/include" "${SHMEM_ROOT}/include")
target_link_directories(ll_combine_bench PRIVATE "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
target_link_libraries(ll_combine_bench PRIVATE MPI::MPI_CXX ascendcl shmem
    tiling_api platform unified_dlog ${CMAKE_DL_LIBS} pthread)
target_compile_definitions(ll_combine_bench PRIVATE DEEPEP_LL_RDMA=1 DEEPEP_LL_METADATA_CORES=4)
add_dependencies(ll_combine_bench ll_bench_variant)
if(DEEPEP_LL_OPS_BASE)
    file(WRITE "${CMAKE_BINARY_DIR}/ops-baseline.txt"
        "implementation=ops-shmem-base\nupstream=${LL_OPS_REV}\ntransport=SHMEM QP Put; quiet at upstream batch boundary\n")
endif()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/bench-build.txt" CONTENT
    "source=${_source_id}\noriginal=${DEEPEP_LL_COMBINE_ORIGINAL}\nbaseline_revision=${LL_COMBINE_BASELINE_REV}\ncombine_source=${DEEPEP_LL_COMBINE_SOURCE}\nsmall_batch=${DEEPEP_LL_COMBINE_SMALL_BATCH}\nrecv_cores=${DEEPEP_LL_COMBINE_RECV_CORES}\nearly_reduce=${DEEPEP_LL_COMBINE_EARLY_REDUCE}\nremote_first=${DEEPEP_LL_COMBINE_REMOTE_FIRST}\ndefer_return=${DEEPEP_LL_COMBINE_DEFER_RETURN}\nprofile=${DEEPEP_LL_COMBINE_PROFILE}\nmpi=${MPI_CXX_COMPILER}\nshmem=${SHMEM_ROOT}\n")
