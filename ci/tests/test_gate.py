# Copyright (c) 2026, Lu Lu
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[2]
runner = runpy.run_path(str(ROOT / "ci/run_a5.py"))


class GateTests(unittest.TestCase):
    def test_device_and_port_configuration(self):
        good = dict(
            DEEPEP_CI_DEVICES="0,1", DEEPEP_CI_MASTER_PORT="19090", DEEPEP_CI_SHMEM_PORT="19091", CI_MODE="quick"
        )
        self.assertEqual(runner["configuration"](good), (2, [19090, 19091]))
        for key, value in (
            ("DEEPEP_CI_DEVICES", ""),
            ("DEEPEP_CI_DEVICES", "0,00"),
            ("DEEPEP_CI_DEVICES", "0,1,2"),
            ("DEEPEP_CI_DEVICES", "0,x"),
            ("DEEPEP_CI_MASTER_PORT", "19091"),
            ("DEEPEP_CI_MASTER_PORT", "1"),
            ("CI_MODE", "scoped"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                runner["configuration"](good | {key: value})

    def test_matrix_checks_real_dispatch_and_weight_channels(self):
        for world in (2, 4, 8):
            for mode, tokens in (("quick", "33"), ("full", "257")):
                cases = runner["cases"](world, mode)
                self.assertEqual(len(cases), 8 if world == 8 else 4)
                for _, script, flags in cases:
                    self.assertEqual(flags[flags.index("--tokens") + 1], tokens)
                    self.assertIn("--route", flags)
                    self.assertIn("all", flags)
                    if script == "test_combine.py":
                        self.assertIn("--with-dispatch", flags)
                self.assertTrue(any("--with-weights" in flags for _, _, flags in cases))
                self.assertTrue(any("--without-weights" in flags for _, _, flags in cases))
                self.assertEqual(any("--rank-dedup" in flags for _, _, flags in cases), world == 8)

    def test_rank_reports_fail_closed(self):
        with tempfile.TemporaryDirectory() as location:
            directory = Path(location)
            data = dict(
                world_size=2, passed=True, checks=3, source_sha256="sha", dispatch_checked=True, notify_checked=True
            )
            for rank in range(2):
                (directory / f"rank-{rank}.json").write_text(json.dumps(data | {"rank": rank}))
            verify = runner["verify_ranks"]
            verify(directory, 2, "sha", combine=True)
            for mutation in (
                {"passed": False},
                {"checks": 0},
                {"checks": True},
                {"source_sha256": "old"},
                {"dispatch_checked": False},
                {"rank": 1},
                {"world_size": 8},
            ):
                with self.subTest(mutation=mutation):
                    (directory / "rank-0.json").write_text(json.dumps(data | {"rank": 0} | mutation))
                    with self.assertRaises(ValueError):
                        verify(directory, 2, "sha", combine=True)
            (directory / "rank-1.json").unlink()
            with self.assertRaises(ValueError):
                verify(directory, 2, "sha")

    def test_cleanup_finds_nested_and_adopted_children(self):
        self.assertEqual(runner["descendants"]({2: 1, 3: 2, 4: 1, 5: 99, 6: 3}, 1), {2, 3, 4, 6})

    def test_timeout_also_reclaims_nested_sessions(self):
        run = runner["run"]
        process = mock.Mock()
        process.wait.side_effect = TimeoutError("timed out")
        group, children = mock.Mock(), mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(run.__globals__, stop_group=group, stop_children=children):
                with mock.patch.object(run.__globals__["subprocess"], "Popen", return_value=process):
                    with self.assertRaises(TimeoutError):
                        run(["test"], cwd=directory, env={}, log=Path(directory) / "test.log", timeout=1)
        group.assert_called_once_with(process)
        children.assert_called_once_with()

    def test_workflow_trust_and_finalization(self):
        workflows = ROOT / ".github/workflows"

        def read(name):
            return yaml.load((workflows / name).read_text(), Loader=yaml.BaseLoader)

        workflow = read("npu-ci.yml")
        self.assertEqual(set(workflow["on"]), {"workflow_dispatch"})
        jobs = workflow["jobs"]
        self.assertEqual(jobs["a5"]["permissions"], {"contents": "read"})
        self.assertIn("always()", jobs["finalize"]["if"])
        self.assertEqual(jobs["finalize"]["runs-on"], "ubuntu-24.04")
        self.assertEqual(workflow["concurrency"]["cancel-in-progress"], "false")
        self.assertEqual(jobs["a5"]["steps"][1]["with"]["fetch-depth"], "0")
        for job in jobs.values():
            for step in job["steps"]:
                if step.get("uses", "").startswith("actions/checkout"):
                    self.assertEqual(step["with"]["persist-credentials"], "false")
        pending = read("npu-status.yml")
        self.assertIn("pull_request_target", pending["on"])
        self.assertIn("default_branch", pending["jobs"]["pending"]["if"])
        checkout = pending["jobs"]["pending"]["steps"][0]["with"]
        self.assertEqual(checkout, {"ref": "${{ github.workflow_sha }}", "persist-credentials": "false"})
        for name in ("cpu-ci.yml", "ci-contract.yml"):
            automatic = read(name)
            self.assertIn("pull_request", automatic["on"])
            self.assertFalse(
                any("paths" in trigger for trigger in automatic["on"].values() if isinstance(trigger, dict))
            )


if __name__ == "__main__":
    unittest.main()
