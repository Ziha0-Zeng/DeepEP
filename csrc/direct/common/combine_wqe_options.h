#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ascend_deepep::udma {

struct CombineWqeOptions {
    uint32_t batch = 32;
    uint32_t direct = 0;
};

inline CombineWqeOptions ReadCombineWqeOptions()
{
    CombineWqeOptions result;
    const char* batch = std::getenv("DEEPEP_CLOS_COMBINE_WQE_BATCH");
    if (batch && *batch) {
        if (std::strcmp(batch, "32") == 0) result.batch = 32;
        else if (std::strcmp(batch, "64") == 0) result.batch = 64;
        else if (std::strcmp(batch, "128") == 0) result.batch = 128;
        else if (std::strcmp(batch, "256") == 0) result.batch = 256;
        else throw std::invalid_argument("DEEPEP_CLOS_COMBINE_WQE_BATCH must be 32/64/128/256");
    }
    const char* builder = std::getenv("DEEPEP_CLOS_COMBINE_WQE_BUILDER");
    if (builder && *builder) {
        if (std::strcmp(builder, "direct") == 0) result.direct = 1;
        else if (std::strcmp(builder, "legacy") != 0)
            throw std::invalid_argument("DEEPEP_CLOS_COMBINE_WQE_BUILDER must be legacy/direct");
    }
    return result;
}

} // namespace ascend_deepep::udma
