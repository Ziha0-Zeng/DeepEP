# Fetch only the kernel dependencies. No ACLNN package or HCCL Host code is built.
set(LL_OPS_REV "e6c7d2e57dd222e5a774ba3c89392a50a6a26a78")
find_package(Git REQUIRED)
set(LL_OPS_SOURCE_DIR "" CACHE PATH "Existing clean ops-transformer checkout at the pinned revision")
if(NOT LL_OPS_SOURCE_DIR)
    set(LL_OPS_SOURCE_DIR "${DEEPEP_SOURCE_DIR}/build/ops-transformer-${LL_OPS_REV}")
    if(NOT EXISTS "${LL_OPS_SOURCE_DIR}/.git")
        file(MAKE_DIRECTORY "${LL_OPS_SOURCE_DIR}")
        execute_process(COMMAND "${GIT_EXECUTABLE}" init "${LL_OPS_SOURCE_DIR}" COMMAND_ERROR_IS_FATAL ANY)
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LL_OPS_SOURCE_DIR}" cat-file -e "${LL_OPS_REV}^{commit}"
        RESULT_VARIABLE _have OUTPUT_QUIET ERROR_QUIET)
    if(NOT _have EQUAL 0)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LL_OPS_SOURCE_DIR}" config remote.ops.url
            https://gitcode.com/zzg_code/ops-transformer.git COMMAND_ERROR_IS_FATAL ANY)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LL_OPS_SOURCE_DIR}" config remote.ops.promisor true
            COMMAND_ERROR_IS_FATAL ANY)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LL_OPS_SOURCE_DIR}" config remote.ops.partialclonefilter blob:none
            COMMAND_ERROR_IS_FATAL ANY)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LL_OPS_SOURCE_DIR}" fetch --depth=1 --filter=blob:none
            ops "${LL_OPS_REV}" COMMAND_ERROR_IS_FATAL ANY)
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LL_OPS_SOURCE_DIR}" sparse-checkout set
        mc2/common/op_kernel mc2/moe_distribute_combine_v2/op_kernel mc2/moe_distribute_dispatch_v2/op_kernel
        COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LL_OPS_SOURCE_DIR}" checkout --detach "${LL_OPS_REV}"
        COMMAND_ERROR_IS_FATAL ANY)
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LL_OPS_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE _rev OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LL_OPS_SOURCE_DIR}" status --porcelain --untracked-files=no
    OUTPUT_VARIABLE _dirty COMMAND_ERROR_IS_FATAL ANY)
if(NOT _rev STREQUAL LL_OPS_REV OR NOT _dirty STREQUAL "")
    message(FATAL_ERROR "ops-shmem-base requires a clean upstream checkout at ${LL_OPS_REV}")
endif()
set(LL_OPS_KERNEL_DIR "${CMAKE_BINARY_DIR}/ops-kernel")
file(MAKE_DIRECTORY "${LL_OPS_KERNEL_DIR}")
file(COPY "${LL_OPS_SOURCE_DIR}/mc2" DESTINATION "${LL_OPS_KERNEL_DIR}")
file(COPY "${LL_OPS_SOURCE_DIR}/LICENSE" DESTINATION "${LL_OPS_KERNEL_DIR}")
set(_kernel "${LL_OPS_KERNEL_DIR}/mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_roce.h")
file(READ "${_kernel}" _text)
# Only the communication adapter changes. Keep partitioning, packet formats,
# ready polling, partial reduction and final reduction from the pinned source.
string(REPLACE "#include \"adv_api/hcomm/hcomm.h\"" "#include \"transport.hpp\"" _text "${_text}")
string(REPLACE "#include \"../../../common/op_kernel/mc2_roce_wqe_config.h\"" "" _text "${_text}")
string(REPLACE "Hcomm<CommProtocol::COMM_PROTOCOL_ROCE>" "opsbench::Transport" _text "${_text}")
string(REPLACE "BatchHandle<ChannelHandle>" "opsbench::Batch" _text "${_text}")
string(REPLACE "const ChannelHandle channel" "const uint64_t channel" _text "${_text}")
string(REPLACE "hcomm_.template WriteNbi<Mc2RoceSend::NO_CQE_CONFIG>" "hcomm_.WriteNbi" _text "${_text}")
file(WRITE "${_kernel}" "${_text}")
message(STATUS "ops-shmem-base: ${LL_OPS_REV}; SHMEM transport adaptation, no ACLNN/HCCL initialization")
