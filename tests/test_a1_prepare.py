"""Offline preparation contracts: synthetic sources/builders, no native run."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sim import a1_prepare as prepare_module
from sim.a1_skills import SKILL_IDS


def synthetic_arrays():
    return {"qpos": np.zeros((2, 7)), "time_s": np.array([0., 2.]),
            "knot_qpos": np.zeros((2, 7)), "knot_time_s": np.array([0., 2.]),
            "coefficients": np.zeros((1, 6, 7))}


def expected_identity():
    arrays = synthetic_arrays()
    identity = {key: {"dtype": a.dtype.str, "shape": list(a.shape),
                     "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
                for key, a in arrays.items()}
    return {name: {"duration_s": 2., "arrays": copy.deepcopy(identity)} for name in SKILL_IDS}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        self.output = self.root / "fresh"
        self.config = {"sources": {}}
        for name in ("ready", "stab", "slash_left", "parry_left", "parry_right"):
            content = ("synthetic-numeric-source-" + name).encode()
            filename = f"{name}.npy"
            (self.sources / filename).write_bytes(content)
            self.config["sources"][name] = {"filename": filename,
                "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        self.reference = self.root / "reference.json"
        self.reference.write_text(json.dumps({"schema_version": 1, "skills": expected_identity()}), encoding="utf-8")
        self.pilot = SimpleNamespace(metadata={"skills": {name: {
            "geometry_candidate_only": True, "all_authored_rows_retained": True,
            "ik": {"success": True}, "strict_static_and_interpolated_screen": {"passed": True}}
            for name in SKILL_IDS}})
        self.bank = SimpleNamespace(reports={"library": {"all_skills_pass_geometry": True}})
        self.calls = []

    def fake_export(self, bank, folder):
        folder.mkdir()
        for name in SKILL_IDS:
            np.savez_compressed(folder / f"{name}.npz", **synthetic_arrays())
        target = folder / ("pilot.json" if bank is self.pilot else "bank.json")
        target.write_text("{}\n", encoding="utf-8")
        self.calls.append(("export", folder.name))
        return target

    def patches(self, *, adapted=None, defence_effect=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(prepare_module, "load_config", return_value=self.config))
        stack.enter_context(patch.object(prepare_module, "_freeze_inputs", return_value={"implementation_sha256": {"fake": "hash"}, "fixed_input_sha256": {}}))
        stack.enter_context(patch.object(prepare_module, "REFERENCE", self.reference))
        pilot = stack.enter_context(patch.object(prepare_module, "build_pilot", return_value=self.pilot))
        adapt = stack.enter_context(patch.object(prepare_module, "build_adapted_a1_library", return_value=adapted or self.bank))
        defend = stack.enter_context(patch.object(prepare_module, "build_defence_candidate",
                                                 side_effect=defence_effect, return_value=self.bank))
        stack.enter_context(patch.object(prepare_module, "export_pilot", side_effect=self.fake_export))
        stack.enter_context(patch.object(prepare_module, "export_a1_library", side_effect=self.fake_export))
        return pilot, adapt, defend

    def test_complete_fixed_pipeline_and_numeric_identity_without_source_mutation(self):
        before = {p.name: p.read_bytes() for p in self.sources.iterdir()}
        pilot, adapt, defend = self.patches()
        report = prepare_module.prepare(self.sources, self.output)
        pilot.assert_called_once_with(self.config, source_dirs=[self.sources], with_ik=True)
        adapt.assert_called_once_with(self.output / "pilot", {"source_time_scale": 4.}, {"amplitude": .8})
        self.assertEqual([c.args[1]["skill_id"] for c in defend.call_args_list], ["parry_right", "parry_left"])
        right = defend.call_args_list[0].args[1]
        self.assertEqual(next(p["duration_s"] for p in right["keyposes"] if p["name"] == "finite_protective_hold"), 8)
        self.assertEqual(self.calls, [("export", n) for n in ("pilot", "adapted", "right25", "library")])
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["inputs_unchanged"])
        self.assertTrue(report["bank_identity"]["numeric_matches_validated_v5"])
        self.assertEqual(report["bank_identity"]["runtime_numeric_identity"], expected_identity())
        self.assertFalse(report["admitted_to_training"])
        self.assertTrue(all(s["status"] == "geometry_passed" for s in report["stages"]))
        self.assertEqual({p.name: p.read_bytes() for p in self.sources.iterdir()}, before)

    def test_missing_size_and_hash_errors_happen_before_output_or_build(self):
        pilot, _, _ = self.patches()
        path = self.sources / "ready.npy"
        original = path.read_bytes()
        for mode, expected in (("missing", FileNotFoundError), ("size", ValueError), ("hash", ValueError)):
            with self.subTest(mode=mode):
                path.write_bytes(original)
                if mode == "missing":
                    path.unlink()
                elif mode == "size":
                    path.write_bytes(original + b"extra")
                else:
                    path.write_bytes(b"x" * len(original))
                with self.assertRaises(expected):
                    prepare_module.prepare(self.sources, self.output)
                self.assertFalse(self.output.exists())
        pilot.assert_not_called()

    def test_failed_geometry_is_exported_and_later_stages_never_run(self):
        failed = SimpleNamespace(reports={"library": {"all_skills_pass_geometry": False},
                                            "failed_rows": [3, 5]})
        _, _, defend = self.patches(adapted=failed)
        with self.assertRaisesRegex(RuntimeError, "adapted: geometry screen failed"):
            prepare_module.prepare(self.sources, self.output)
        report = json.loads((self.output / "prepare_report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stages"][-1]["geometry"]["reports"]["failed_rows"], [3, 5])
        self.assertTrue((self.output / "adapted/bank.json").is_file())
        self.assertFalse((self.output / "right25").exists())
        defend.assert_not_called()

    def test_builder_exception_stops_and_preserves_prior_stage(self):
        _, _, defend = self.patches(defence_effect=RuntimeError("synthetic solver failure"))
        with self.assertRaisesRegex(RuntimeError, "synthetic solver failure"):
            prepare_module.prepare(self.sources, self.output)
        report = json.loads((self.output / "prepare_report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual([s["name"] for s in report["stages"]], ["pilot", "adapted", "right25"])
        self.assertEqual(report["stages"][-1]["status"], "failed")
        self.assertTrue((self.output / "adapted/bank.json").is_file())
        self.assertFalse((self.output / "library").exists())
        self.assertEqual(defend.call_count, 1)

    def test_existing_even_empty_outputs_and_source_subdirectories_are_rejected(self):
        pilot, _, _ = self.patches()
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            prepare_module.prepare(self.sources, self.output)
        self.assertEqual(list(self.output.iterdir()), [])
        with self.assertRaisesRegex(ValueError, "inside the ASE source"):
            prepare_module.prepare(self.sources, self.sources / "new")
        pilot.assert_not_called()

    def test_runtime_array_mismatch_fails_without_inheriting_reference_validation(self):
        self.patches()
        bad = {"schema_version": 1, "skills": expected_identity()}
        bad["skills"]["stab"]["arrays"]["qpos"]["sha256"] = "0" * 64
        self.reference.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "differs from pinned v5"):
            prepare_module.prepare(self.sources, self.output)
        report = json.loads((self.output / "prepare_report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["bank_identity"]["numeric_matches_validated_v5"])
        self.assertTrue((self.output / "library/bank.json").is_file())
        self.assertFalse(report["admitted_to_training"])


if __name__ == "__main__":
    unittest.main()
