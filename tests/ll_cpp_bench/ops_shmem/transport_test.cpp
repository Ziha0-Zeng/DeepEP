#include <cstdint>
#define GM_ADDR uint8_t*
#include "transport.hpp"
#include <cassert>
#include <iostream>

int main()
{
    uint8_t local[8192]{}, remote[8192]{}, scratch[512]{};
    Model::local = local;
    Model::remote = remote;
    AscendC::core = 53;
    std::fill(local + 512, local + 1024, 0x21);
    std::fill(local + 1024, local + 1536, 0x43);
    opsbench::Transport transport;
    transport.Init({scratch}, sizeof(scratch));
    auto batch = transport.MakeBatchHandle(17, {scratch}, sizeof(scratch), local, local);
    transport.WriteNbi(batch, local + 4096, local + 512, 512);
    transport.WriteNbi(batch, local + 4608, local + 1024, 512);
    assert(Model::pending.size() == 2 && Model::quiets == 0);
    for (const auto& request : Model::pending) {
        assert(request.peer == 17 && request.qp == 53);
        assert(request.src >= local + 512 && request.src < local + 1536);
        assert(request.dst >= remote + 4096 && request.dst < remote + 5120);
    }
    transport.BatchCommit(batch);
    assert(Model::pending.empty() && Model::quiets == 1);
    assert(std::memcmp(remote + 4096, local + 512, 1024) == 0);
    transport.BatchCommit(batch);
    assert(Model::quiets == 1);  // empty batch has no completion to retire
    std::cout << "SHMEM adapter: symmetric translation, per-core QP and batch completion passed\n";
}
