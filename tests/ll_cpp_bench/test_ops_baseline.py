"""Metadata/transport adaptation checks. Device execution requires CANN/NPU."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

HERE = Path(__file__).resolve().parent


def test_ops_fixture_routes_and_gather_masks(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG")
    exe = tmp_path / ("fixture.exe" if os.name == "nt" else "fixture")
    subprocess.run([zig, "c++", "-std=c++17", "-O2", "-UNDEBUG",
                    str(HERE / "ops_shmem/fixture_test.cpp"), "-o", str(exe)], check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=180)


def test_ops_shmem_transport(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG")
    exe = tmp_path / ("transport.exe" if os.name == "nt" else "transport")
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG",
                    "-I", str(HERE.parent / "ll_rdma/stubs"),
                    str(HERE / "ops_shmem/transport_test.cpp"), "-o", str(exe)], check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)


def test_pinned_ops_adaptation_changes_only_transport(tmp_path):
    source = os.environ.get("DEEPEP_TEST_OPS_SOURCE")
    cmake = shutil.which("cmake")
    if not source or not cmake:
        pytest.skip("Set DEEPEP_TEST_OPS_SOURCE to the pinned upstream checkout")
    source = Path(source)
    wrapper = tmp_path / "source.cmake"
    wrapper.write_text(f'set(DEEPEP_SOURCE_DIR "{HERE.parents[1].as_posix()}")\n'
                       f'set(LL_OPS_SOURCE_DIR "{source.as_posix()}" CACHE PATH "")\n'
                       f'include("{(HERE / "ops_shmem/source.cmake").as_posix()}")\n')
    subprocess.run([cmake, "-P", str(wrapper)], cwd=tmp_path, check=True, timeout=60)
    rel = Path("mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_roce.h")
    before = (source / rel).read_text(encoding="utf-8")
    after = (tmp_path / "ops-kernel" / rel).read_text(encoding="utf-8")
    replacements = (
        ('#include "adv_api/hcomm/hcomm.h"', '#include "transport.hpp"'),
        ('#include "../../../common/op_kernel/mc2_roce_wqe_config.h"', ''),
        ('Hcomm<CommProtocol::COMM_PROTOCOL_ROCE>', 'opsbench::Transport'),
        ('BatchHandle<ChannelHandle>', 'opsbench::Batch'),
        ('const ChannelHandle channel', 'const uint64_t channel'),
        ('hcomm_.template WriteNbi<Mc2RoceSend::NO_CQE_CONFIG>', 'hcomm_.WriteNbi'),
    )
    expected = before
    for old, new in replacements:
        assert old in expected
        expected = expected.replace(old, new)
    assert after == expected
    assert (tmp_path / "ops-kernel/LICENSE").read_bytes() == (source / "LICENSE").read_bytes()
